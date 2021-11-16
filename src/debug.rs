use ash::vk;
use log::{trace, debug, warn, error, info};

use std::os::raw::c_void;
use std::ffi::CStr;
use std::ptr;

use super::vk_to_string;

pub struct ValidationInfo {
    pub is_enabled: bool,
    pub required_validation_layers: [&'static str; 1],
}

pub const VALIDATION: ValidationInfo = ValidationInfo {
    is_enabled: true,
    required_validation_layers: ["VK_LAYER_KHRONOS_validation"],
};

unsafe extern "system" fn vulkan_debug_utils_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut c_void,
) -> vk::Bool32 {
    let types = match message_type {
        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => "[General]",
        vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "[Performance]",
        vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "[Validation]",
        _ => "[Unknown]",
    };
    let message = CStr::from_ptr((*p_callback_data).p_message);
    match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => debug!("[Verbose]{}{:?}", types, message),
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => warn!("[Warning]{}{:?}", types, message),
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => error!("[Error]{}{:?}", types, message),
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => info!("[Info]{}{:?}", types, message),
        _ => trace!("[Unknown]{}{:?}", types, message),
    };

    vk::FALSE
}

pub fn setup_debug_utils(
    entry: &ash::Entry,
    instance: &ash::Instance,
) -> (ash::extensions::ext::DebugUtils, vk::DebugUtilsMessengerEXT) {
    let debug_utils_loader = ash::extensions::ext::DebugUtils::new(entry, instance);

    if VALIDATION.is_enabled {
        let messenger_ci = populate_debug_messenger_create_info();

        let utils_messenger = unsafe {
            debug_utils_loader
                .create_debug_utils_messenger(&messenger_ci, None)
                .expect("Debug utils callback")
        };
        (debug_utils_loader, utils_messenger)
    } else {
        (debug_utils_loader, ash::vk::DebugUtilsMessengerEXT::null())
    }
}


pub fn check_validation_layer_support(entry: &ash::Entry) -> bool {
    let layer_properties = entry
        .enumerate_instance_layer_properties()
        .expect("Failed to enumerate instance layer properties");

    if layer_properties.is_empty() {
        error!("No available layers");
        return false;
    } else  {
        trace!("Available layers:");
        for layer in layer_properties.iter() {
            let layer_name = vk_to_string(&layer.layer_name);
            trace!("\t{}", layer_name);
        }
    }

    for required_layer_name in VALIDATION.required_validation_layers.iter() {
        for layer_property in layer_properties.iter() {
            let test_layer_name = vk_to_string(&layer_property.layer_name);
            if (*required_layer_name) == test_layer_name {
                return true;
            }
        }
    }

    false
}

pub fn populate_debug_messenger_create_info() -> vk::DebugUtilsMessengerCreateInfoEXT {
    vk::DebugUtilsMessengerCreateInfoEXT {
        s_type: vk::StructureType::DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        p_next: ptr::null(),
        flags: vk::DebugUtilsMessengerCreateFlagsEXT::empty(),
        message_severity: vk::DebugUtilsMessageSeverityFlagsEXT::WARNING |
        // vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE |
        // vk::DebugUtilsMessageSeverityFlagsEXT::INFO |
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
        message_type: vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
            | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
            | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
        pfn_user_callback: Some(vulkan_debug_utils_callback),
        p_user_data: ptr::null_mut(),
    }
}
