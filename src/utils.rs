use ash::vk;
use ash::vk::{api_version_major, api_version_minor, api_version_patch};
use gpu_allocator::vulkan::*;

use std::collections::HashSet;
use std::ffi::CString;
use std::os::raw::c_void;
use std::ptr;

use super::debug;
use super::errors::{Error, Result};
use super::platforms;
use super::vk_to_string;

pub fn find_depth_format(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<vk::Format> {
    find_supported_format(
        instance,
        physical_device,
        &[
            vk::Format::D32_SFLOAT,
            vk::Format::D32_SFLOAT_S8_UINT,
            vk::Format::D24_UNORM_S8_UINT,
        ],
        vk::ImageTiling::OPTIMAL,
        vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
    )
}

pub fn find_supported_format(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    candidate_formats: &[vk::Format],
    tiling: vk::ImageTiling,
    features: vk::FormatFeatureFlags,
) -> Result<vk::Format> {
    for &format in candidate_formats.iter() {
        let format_properties =
            unsafe { instance.get_physical_device_format_properties(physical_device, format) };
        let linear_acceptable = tiling == vk::ImageTiling::LINEAR
            && format_properties.linear_tiling_features.contains(features);
        let optimal_acceptable = tiling == vk::ImageTiling::OPTIMAL
            && format_properties.optimal_tiling_features.contains(features);
        if linear_acceptable || optimal_acceptable {
            return Ok(format);
        }
    }

    Err(Error::NoFormatsAvailable{
        tiling,
        features,
    })
}

pub fn has_required_queues(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    surface_loader: &ash::extensions::khr::Surface,
    surface: vk::SurfaceKHR,
) -> Result<bool> {
    let queue_families = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
    let mut has_graphics_queue = false;
    let mut has_presentation_queue = false;

    let mut index = 0;
    for queue_family in queue_families.iter() {
        if queue_family.queue_count > 0 && queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
            has_graphics_queue = true;
        }

        let is_present_supported = unsafe {
            Error::wrap_result(
                surface_loader
                    .get_physical_device_surface_support(
                        physical_device,
                        index as u32,
                        surface,
                    ),
                "Failed to query present support",
            )?
        };

        if queue_family.queue_count > 0 && is_present_supported {
            has_presentation_queue = true;
        }

        if has_graphics_queue && has_presentation_queue {
            return Ok(true);
        }

        index += 1;
    }
    Ok(false)
}

pub fn check_device_extension_support(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    required_extensions_list: &[String],
) -> Result<bool> {
    let available_extensions = unsafe {
        Error::wrap_result(
            instance
                .enumerate_device_extension_properties(physical_device),
            "Failed to get device extension properties",
        )?
    };

    let mut available_extension_names = vec![];

    println!("\tAvailable device extensions:");
    for extension in available_extensions.iter() {
        let extension_name = vk_to_string(&extension.extension_name);
        println!(
            "\t\tName: {}, Version: {}",
            extension_name, extension.spec_version,
        );

        available_extension_names.push(extension_name);
    }

    let mut required_extensions: HashSet<String> = HashSet::new();
    for extension in required_extensions_list.iter() {
        required_extensions.insert(extension.to_string());
    }

    for extension_name in available_extension_names.iter() {
        required_extensions.remove(extension_name);
    }

    Ok(required_extensions.is_empty())
}

pub fn pick_physical_device(
    instance: &ash::Instance,
    surface_loader: &ash::extensions::khr::Surface,
    surface: vk::SurfaceKHR,
    required_extensions: &[String],
) -> Result<vk::PhysicalDevice> {
    let physical_devices = unsafe {
        Error::wrap_result(
            instance.enumerate_physical_devices(),
            "Failed to enumerate physical devices",
        )?
    };
    println!(
        "{} devices (GPU) found with Vulkan support.",
        physical_devices.len()
    );

    for &physical_device in physical_devices.iter() {
        if is_physical_device_suitable(instance, physical_device, surface_loader, surface, required_extensions)? {
            return Ok(physical_device);
        }
    }
    Err(Error::NoSuitableDevices)
}

fn is_physical_device_suitable(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    surface_loader: &ash::extensions::khr::Surface,
    surface: vk::SurfaceKHR,
    required_extensions: &[String],
) -> Result<bool> {
    let device_properties = unsafe{ instance.get_physical_device_properties(physical_device) };
    let device_features = unsafe{ instance.get_physical_device_features(physical_device) };
    let device_queue_families = unsafe{ instance.get_physical_device_queue_family_properties(physical_device) };

    let device_type = match device_properties.device_type {
        vk::PhysicalDeviceType::CPU => "cpu",
        vk::PhysicalDeviceType::INTEGRATED_GPU => "Integrated GPU",
        vk::PhysicalDeviceType::DISCRETE_GPU => "Discrete GPU",
        vk::PhysicalDeviceType::VIRTUAL_GPU => "Virtual GPU",
        vk::PhysicalDeviceType::OTHER => "Unknown",
        _ => "Unknown",
    };

    let device_name = vk_to_string(&device_properties.device_name);
    println!(
        "\tDevice name: {}, id: {}, type: {}",
        device_name, device_properties.device_id, device_type,
    );

    let major_version = api_version_major(device_properties.api_version);
    let minor_version = api_version_minor(device_properties.api_version);
    let patch_version = api_version_patch(device_properties.api_version);

    println!(
        "\tAPI version: {}.{}.{}",
        major_version, minor_version, patch_version,
    );

    println!("\tSupported queue families: {}", device_queue_families.len());
    println!("\t\tQueue count | Graphics, Compute, Transfer, Sparse Binding");
    for queue_family in device_queue_families.iter() {
        let is_graphics_supported = if queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
            "supported"
        } else  {
            "unsupported"
        };
        let is_compute_supported = if queue_family.queue_flags.contains(vk::QueueFlags::COMPUTE) {
            "supported"
        } else  {
            "unsupported"
        };
        let is_transfer_supported = if queue_family.queue_flags.contains(vk::QueueFlags::TRANSFER) {
            "supported"
        } else  {
            "unsupported"
        };
        let is_sparse_supported = if queue_family.queue_flags.contains(vk::QueueFlags::SPARSE_BINDING) {
            "supported"
        } else  {
            "unsupported"
        };
        println!(
            "\t\t{}\t    | {},  {},  {},  {}",
            queue_family.queue_count,
            is_graphics_supported,
            is_compute_supported,
            is_transfer_supported,
            is_sparse_supported,
        );
    }

    println!(
        "\tGeometry shader support: {}",
        if device_features.geometry_shader == 1 {
            "yes"
        } else {
            "no"
        },
    );

    if !has_required_queues(instance, physical_device, surface_loader, surface)? {
        return Ok(false);
    }

    let is_device_extension_supported = check_device_extension_support(instance, physical_device, required_extensions)?;
    let is_swapchain_supported = if is_device_extension_supported {
        let swapchain_support = query_swapchain_support(physical_device, surface_loader, surface)?;
        !swapchain_support.formats.is_empty() && !swapchain_support.present_modes.is_empty()
    } else {
        false
    };

    Ok(is_device_extension_supported && is_swapchain_supported)
}


pub struct SwapChainSupport {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

pub fn query_swapchain_support(
    physical_device: vk::PhysicalDevice,
    surface_loader: &ash::extensions::khr::Surface,
    surface: vk::SurfaceKHR,
) -> Result<SwapChainSupport> {
    unsafe {
        let capabilities = Error::wrap_result(
            surface_loader
                .get_physical_device_surface_capabilities(physical_device, surface),
            "Failed to query for surface capabilities.",
        )?;
        let formats = Error::wrap_result(
            surface_loader
                .get_physical_device_surface_formats(physical_device, surface),
            "Failed to query for surface formats.",
        )?;
        let present_modes = Error::wrap_result(
            surface_loader
                .get_physical_device_surface_present_modes(physical_device, surface),
            "Failed to query for surface present mode.",
        )?;

        Ok(SwapChainSupport {
            capabilities,
            formats,
            present_modes,
        })
    }
}

fn try_create_instance(
    entry: &ash::Entry,
    app_info: &vk::ApplicationInfo,
    required_extensions: &[*const i8],
    debug_utils_create_info: &vk::DebugUtilsMessengerCreateInfoEXT,
) -> Result<ash::Instance> {
    let required_validation_layer_raw_names: Vec<CString> = debug::VALIDATION
        .required_validation_layers
        .iter()
        .map(|layer_name| CString::new(*layer_name).unwrap())
        .collect();
    let enable_layer_names: Vec<*const i8> = required_validation_layer_raw_names
        .iter()
        .map(|layer_name| layer_name.as_ptr())
        .collect();

    let create_info = vk::InstanceCreateInfo {
        s_type: vk::StructureType::INSTANCE_CREATE_INFO,
        p_next: if debug::VALIDATION.is_enabled {
            debug_utils_create_info as *const vk::DebugUtilsMessengerCreateInfoEXT
                as *const c_void
        } else {
            ptr::null()
        },
        flags: vk::InstanceCreateFlags::empty(),
        p_application_info: app_info,
        pp_enabled_layer_names: if debug::VALIDATION.is_enabled {
            enable_layer_names.as_ptr()
        } else {
            ptr::null()
        },
        enabled_layer_count: if debug::VALIDATION.is_enabled {
            enable_layer_names.len()
        } else {
            0
        } as u32,
        pp_enabled_extension_names: required_extensions.as_ptr(),
        enabled_extension_count: required_extensions.len() as u32,
    };

    unsafe {
        match entry.create_instance(&create_info, None) {
            Ok(v) => Ok(v),
            Err(ash::InstanceError::LoadError(msgs)) => {
                eprintln!("LoadError:");
                for msg in msgs {
                    eprintln!("\t{}", msg);
                }
                Err(Error::internal("Failed to create instance: load error (see stderr)"))
            },
            Err(ash::InstanceError::VkError(vk_err)) => Err(Error::wrap(vk_err, "Failed to create instance")),
        }
    }
}

pub fn create_instance(
    entry: &ash::Entry,
    app_name: &str,
    engine_name: &str,
    app_version: u32,
    engine_version: u32,
    api_version: u32,
) -> Result<ash::Instance> {
    if debug::VALIDATION.is_enabled && !debug::check_validation_layer_support(entry) {
        panic!("Validation layers requested but not available!");
    }

    let c_app_name = CString::new(app_name).unwrap();
    let c_engine_name = CString::new(engine_name).unwrap();
    let app_info = vk::ApplicationInfo {
        s_type: vk::StructureType::APPLICATION_INFO,
        p_next: ptr::null(),
        p_application_name: c_app_name.as_ptr(),
        application_version: app_version,
        p_engine_name: c_engine_name.as_ptr(),
        engine_version: engine_version,
        api_version: api_version,
    };

    let debug_utils_create_info = debug::populate_debug_messenger_create_info();
    let extension_names = platforms::required_extension_names();
    let mut try_extension_names = vec![];
    for ext in extension_names.iter() {
        try_extension_names.push(*ext);
    }
    for ext in platforms::optional_extension_names() {
        try_extension_names.push(ext);
    }
    let maybe_instance = try_create_instance(
        entry,
        &app_info,
        &try_extension_names,
        &debug_utils_create_info,
    );
    match maybe_instance {
        Ok(instance) => Ok(instance),
        Err(_) => try_create_instance(
            entry,
            &app_info,
            &extension_names,
            &debug_utils_create_info,
        ),
    }
}

pub fn create_allocator(
    instance: &ash::Instance,
    physical_device: &vk::PhysicalDevice,
    device: &ash::Device,
) -> Result<Allocator> {
    let res = Allocator::new(&AllocatorCreateDesc{
        instance: instance.clone(),
        device: device.clone(),
        physical_device: physical_device.clone(),
        debug_settings: Default::default(),
        buffer_device_address: true,
    });
    match res {
        Ok(v) => Ok(v),
        Err(e) => Err(Error::AllocationError(e)),
    }
}
