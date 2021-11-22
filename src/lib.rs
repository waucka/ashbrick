//! A library that makes [`ash`] easier and less error-prone to work with
//!
//! It's kind of like [`Vulkano`], but far less ambitious.
//!
//! [`ash`]: https://crates.io/crates/ash
//! [`Vulkano`]: http://vulkano.rs/

use ash::vk;
use winit::event_loop::EventLoop;

pub use ash;
pub use winit;
#[cfg(feature = "egui")]
pub use egui;
pub use log;

use std::ptr;
use std::cell::RefCell;
use std::ffi::{CString, CStr};
use std::os::raw::{c_char, c_void};
use std::rc::Rc;

use debug::VALIDATION;

pub const MAX_FRAMES_IN_FLIGHT: usize = 2;

mod debug;
mod errors;
mod platforms;
mod utils;
mod window;

pub use errors::{Error, Result};

macro_rules! impl_defaulted_setter {
    ( $fn_name:ident, $field_name:ident, str ) => {
    pub fn $fn_name(mut self, $field_name: &str) -> Self {
        self.$field_name = $field_name.to_string();
        self
    }
    };
    ( $fn_name:ident, $field_name:ident, $type:ty, ref ) => {
    pub fn $fn_name(mut self, $field_name: &$type) -> Self {
        self.$field_name = $field_name;
        self
    }
    };
    ( $fn_name:ident, $field_name:ident, $type:ty) => {
    pub fn $fn_name(mut self, $field_name: $type) -> Self {
        self.$field_name = $field_name;
        self
    }
    };
}

pub mod buffer;
pub mod command_buffer;
pub mod compute;
pub mod image;
pub mod renderer;
pub mod shader;
pub mod sync;
pub mod texture;
pub mod descriptor;

use buffer::HasBuffer;
use command_buffer::{CommandBuffer, CommandPool};

pub fn vk_to_string(raw_string_array: &[c_char]) -> String {
    let raw_string = unsafe {
        let pointer = raw_string_array.as_ptr();
        CStr::from_ptr(pointer)
    };

    raw_string
        .to_str()
        .expect("Failed to convert vulkan raw string.")
        .to_owned()
}

#[derive(Copy, Clone)]
pub struct FrameId {
    idx: usize,
}

impl FrameId {
    fn initial() -> Self {
        Self{
            idx: 0,
        }
    }

    fn advance(&mut self) {
        self.idx = (self.idx + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    pub fn next(&self) -> Self {
        Self {
            idx: (self.idx + 1) % MAX_FRAMES_IN_FLIGHT,
        }
    }
}

impl From<usize> for FrameId {
    fn from(idx: usize) -> Self {
        if idx >= MAX_FRAMES_IN_FLIGHT {
            panic!(
                "Tried to create a FrameId with index {} (must be < {})",
                idx,
                MAX_FRAMES_IN_FLIGHT,
            );
        }
        Self{
            idx,
        }
    }
}

impl From<u32> for FrameId {
    fn from(idx: u32) -> Self {
        if idx as usize >= MAX_FRAMES_IN_FLIGHT {
            panic!(
                "Tried to create a FrameId with index {} (must be < {})",
                idx,
                MAX_FRAMES_IN_FLIGHT,
            );
        }
        Self{
            idx: idx as usize,
        }
    }
}

impl std::fmt::Display for FrameId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.idx.fmt(f)
    }
}

pub struct PerFrameSet<T> {
    items: Vec<T>,
}

impl<T> PerFrameSet<T> {
    pub fn new<F>(mut item_generator: F) -> Result<Self>
    where
        F: FnMut(FrameId) -> Result<T>
    {
        let mut items = Vec::new();
        for i in 0..MAX_FRAMES_IN_FLIGHT {
            items.push(item_generator(FrameId::from(i))?);
        }
        Ok(Self{
            items,
        })
    }

    pub fn get(&self, frame: FrameId) -> &T {
        &self.items[frame.idx]
    }

    pub fn get_mut(&mut self, frame: FrameId) -> &mut T {
        &mut self.items[frame.idx]
    }

    // Extracts data from self and creates a new PerFrameSet containing the extracted data
    pub fn extract<F, R>(&self, extractor: F) -> Result<PerFrameSet<R>>
    where
        F: Fn(&T) -> Result<R>
    {
        let new_set: PerFrameSet<R> = PerFrameSet::new(|frame| {
            extractor(self.get(FrameId::from(frame)))
        })?;
        Ok(new_set)
    }

    pub fn foreach<F>(&self, mut action: F) -> Result<()>
    where
        F: FnMut(FrameId, &T) -> Result<()>
    {
        for i in 0..MAX_FRAMES_IN_FLIGHT {
            let item = &self.items[i];
            action(FrameId::from(i), item)?;
        }
        Ok(())
    }

    pub fn foreach_mut<F>(&mut self, mut action: F) -> Result<()>
    where
        F: FnMut(FrameId, &mut T) -> Result<()>
    {
        for i in 0..MAX_FRAMES_IN_FLIGHT {
            let item = &mut self.items[i];
            action(FrameId::from(i), item)?;
        }
        Ok(())
    }

    pub fn replace<F>(&mut self, mut constructor: F) -> Result<()>
    where
        F: FnMut(FrameId, &T) -> Result<T>
    {
        let mut new_items = Vec::new();
        for i in 0..MAX_FRAMES_IN_FLIGHT {
            let old_item = &self.items[i];
            new_items.push(constructor(FrameId::from(i), old_item)?);
        }
        self.items = new_items;
        Ok(())
    }
}

impl<T: Clone> Clone for PerFrameSet<T> {
    fn clone(&self) -> Self {
        Self{
            items: self.items.clone(),
        }
    }
}

// Device

pub const ENGINE_NAME: &'static str = "Wanderer Engine";
pub const ENGINE_VERSION: u32 = vk::make_api_version(0, 0, 1, 0);
pub const VULKAN_API_VERSION: u32 = vk::make_api_version(0, 1, 2, 131);

#[derive(Copy, Clone)]
pub struct Queue {
    family_idx: u32,
    queue_idx: u32,
    flags: vk::QueueFlags,
    can_present: bool,
    queue: vk::Queue,
}

impl Queue {
    fn new(
        device: Rc<InnerDevice>,
        family_idx: u32,
        queue_idx: u32,
        flags: vk::QueueFlags,
        can_present: bool,
    ) -> Result<Self> {
        let queue = unsafe {
            device.device.get_device_queue(family_idx, queue_idx)
        };

        Ok(Self{
            family_idx,
            queue_idx,
            flags,
            can_present,
            queue,
        })
    }

    pub fn get_family(&self) -> QueueFamilyRef {
        QueueFamilyRef{
            idx: self.family_idx,
        }
    }

    pub fn can_do_graphics(&self) -> bool {
        self.flags.contains(vk::QueueFlags::GRAPHICS)
    }

    pub fn can_present(&self) -> bool {
        self.can_present
    }

    pub fn can_do_compute(&self) -> bool {
        self.flags.contains(vk::QueueFlags::COMPUTE)
    }

    pub fn can_do_transfer(&self) -> bool {
        self.flags.contains(vk::QueueFlags::TRANSFER)
    }

    pub fn can_do_sparse_binding(&self) -> bool {
        self.flags.contains(vk::QueueFlags::SPARSE_BINDING)
    }

    fn get(&self) -> vk::Queue {
        self.queue
    }
}

impl std::cmp::PartialEq for Queue {
    fn eq(&self, other: &Self) -> bool {
        self.family_idx == other.family_idx && self.queue_idx == other.queue_idx
    }
}
impl std::cmp::Eq for Queue {}

pub struct DeviceBuilder {
    window_title: String,
    application_version: u32,
    window_size: (u32, u32),
    extensions: Vec<String>,
    validation_enabled: bool,
    windowing_prefs: platforms::WindowingPreferences,
    features_chain: *mut c_void,
}

impl DeviceBuilder {
    pub fn new() -> Self {
        Self {
            window_title: "Some Random Application".to_string(),
            application_version: vk::make_api_version(0, 0, 1, 0),
            window_size: (640, 480),
            extensions: Vec::new(),
            validation_enabled: false,
            windowing_prefs: Default::default(),
            features_chain: ptr::null_mut(),
        }
    }

    pub fn get_extensions(&self) -> &[String] {
        &self.extensions
    }

    impl_defaulted_setter!(with_window_title, window_title, str);
    impl_defaulted_setter!(with_validation, validation_enabled, bool);

    pub fn with_application_version(mut self, major: u32, minor: u32, patch: u32) -> Self {
        self.application_version = vk::make_api_version(0, major, minor, patch);
        self
    }

    pub fn with_window_size(mut self, width: usize, height: usize) -> Self {
        self.window_size = (width as u32, height as u32);
        self
    }

    pub fn with_default_extensions(mut self) -> Self {
        self.extensions.push("VK_KHR_swapchain".to_string());
        self.extensions.push("VK_EXT_descriptor_indexing".to_string());
        self
    }

    pub fn with_extension(mut self, extension_name: &str) -> Self {
        self.extensions.push(extension_name.to_string());
        self
    }

    pub fn with_windowing_prefs(mut self, prefs: platforms::WindowingPreferences) -> Self {
        self.windowing_prefs = prefs;
        self
    }

    pub fn with_features_chain(mut self, features_chain: *mut c_void) -> Self {
        self.features_chain = features_chain;
        self
    }
}

pub struct Device {
    inner: Rc<InnerDevice>,
}

impl Device {
    pub fn new(event_loop: &EventLoop<()>, builder: DeviceBuilder) -> Result<Self> {
        Ok(Self {
            inner: InnerDevice::new(event_loop, builder)?,
        })
    }

    pub fn check_mipmap_support(
        &self,
        image_format: vk::Format,
    ) -> Result<()>{
        let format_properties = unsafe {
            self.inner.instance.get_physical_device_format_properties(self.inner.physical_device, image_format)
        };

        let is_sample_image_filter_linear_supported = format_properties
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR);
        if !is_sample_image_filter_linear_supported {
            Err(Error::Generic("Texture image format does not support linear filtering".to_owned()))
        } else {
            Ok(())
        }
    }

    pub fn get_max_usable_sample_count(
        &self,
    ) -> vk::SampleCountFlags {
        let physical_device_properties =
            unsafe { self.inner.instance.get_physical_device_properties(self.inner.physical_device) };
        let count = std::cmp::min(
            physical_device_properties
                .limits
                .framebuffer_color_sample_counts,
            physical_device_properties
                .limits
                .framebuffer_depth_sample_counts,
        );

        if count.contains(vk::SampleCountFlags::TYPE_64) {
            return vk::SampleCountFlags::TYPE_64;
        }
        if count.contains(vk::SampleCountFlags::TYPE_32) {
            return vk::SampleCountFlags::TYPE_32;
        }
        if count.contains(vk::SampleCountFlags::TYPE_16) {
            return vk::SampleCountFlags::TYPE_16;
        }
        if count.contains(vk::SampleCountFlags::TYPE_8) {
            return vk::SampleCountFlags::TYPE_8;
        }
        if count.contains(vk::SampleCountFlags::TYPE_4) {
            return vk::SampleCountFlags::TYPE_4;
        }
        if count.contains(vk::SampleCountFlags::TYPE_2) {
            return vk::SampleCountFlags::TYPE_2;
        }

        vk::SampleCountFlags::TYPE_1
    }

    pub fn wait_for_idle(&self) -> Result<()>{
        unsafe {
            Error::wrap_result(
                self.inner.device
                    .device_wait_idle(),
                "Failed to wait for device idle",
            )?
        }
        Ok(())
    }

    pub fn window_ref(&self) -> &winit::window::Window {
        &self.inner.window
    }

    pub fn get_queue_families(&self) -> Vec<QueueFamily> {
        self.inner.queue_families.clone()
    }

    // A "super" queue is a queue that can do all of the following:
    //  - present
    //  - compute
    //  - graphics
    //  - transfer
    // In the real world, it looks like all hardware that matters
    // has at least one of these.  Some appear to have no queues
    // that support transfer; according to some guy on the Internet,
    // those queues actually do support transfer.
    // Some devices don't seem to report any queues that can present.
    // I submit that those drivers are written by crack-addled doofuses.
    // Nonetheless, I will support them by falling back to a queue that
    // can at least do graphics and compute, assuming that it can also
    // do transfer and presentation even if it says it can't.
    pub fn get_super_queue_family(&self) -> QueueFamily {
        self.inner.super_queue_family
    }

    pub fn get_window_size(&self) -> (usize, usize) {
        let size = self.inner.window.inner_size();
        (size.width as usize, size.height as usize)
    }

    pub fn get_float_target_format(&self) -> Result<vk::Format> {
        let candidate_formats = [
                vk::Format::R16G16B16_SFLOAT,
                vk::Format::R16G16B16A16_SFLOAT,
                vk::Format::R32G32B32_SFLOAT,
                vk::Format::R32G32B32A32_SFLOAT,
        ];

        for candidate in &candidate_formats {
            let ok = unsafe {
                let res = self.inner.instance.get_physical_device_image_format_properties(
                    self.inner.physical_device,
                    *candidate,
                    vk::ImageType::TYPE_2D,
                    vk::ImageTiling::OPTIMAL,
                    vk::ImageUsageFlags::COLOR_ATTACHMENT,
                    vk::ImageCreateFlags::empty(),
                );
                match res {
                    Ok(_) => true,
                    Err(_) => false,
                }
            };
            if ok {
                return Ok(*candidate)
            }
        }
        Err(Error::Generic("Failed to find a suitable float target format".to_owned()))
    }

    pub fn get_swapchain_image_formats(&self) -> Result<(vk::Format, vk::Format)> {
        let swapchain_support = utils::query_swapchain_support(
            self.inner.physical_device,
            &self.inner.surface_loader,
            self.inner.surface,
        )?;
        let surface_format = utils::choose_swapchain_format(&swapchain_support.formats);
        let depth_format = utils::find_depth_format(
            &self.inner.instance,
            self.inner.physical_device,
        )?;
        Ok((surface_format.format, depth_format))
    }

    pub fn transfer_buffer<T: HasBuffer>(
        &self,
        buffer: T,
        src: Queue,
        src_stage_flags: vk::PipelineStageFlags,
        src_access_mask: vk::AccessFlags,
        dst: Queue,
        dst_stage_flags: vk::PipelineStageFlags,
        dst_access_mask: vk::AccessFlags,
        deps: vk::DependencyFlags,
        pool: Rc<CommandPool>,
        queue: &Queue,
    ) -> Result<()> {
        if src.family_idx == dst.family_idx {
            // No transfer needed!
            return Ok(());
        }
        let buf = buffer.get_buffer();
        let size = buffer.get_size();
        CommandBuffer::run_oneshot_internal(
            Rc::clone(&self.inner),
            pool,
            queue,
            |writer| {
                writer.pipeline_barrier(
                    src_stage_flags,
                    dst_stage_flags,
                    deps,
                    // No generic memory barriers needed for this
                    &[],
                    &[vk::BufferMemoryBarrier{
                        s_type: vk::StructureType::BUFFER_MEMORY_BARRIER,
                        p_next: ptr::null(),
                        src_access_mask,
                        dst_access_mask,
                        src_queue_family_index: src.family_idx,
                        dst_queue_family_index: dst.family_idx,
                        buffer: buf,
                        offset: 0,
                        size: size,
                    }],
                    // No image memory barriers needed for this, obviously
                    &[],
                );
                Ok(())
            }
        )
    }
}

pub enum MemoryUsage {
    Unknown,
    GpuOnly,
    CpuToGpu,
    GpuToCpu,
}

impl MemoryUsage {
    fn as_gpu_allocator(self) -> gpu_allocator::MemoryLocation {
        match self {
            MemoryUsage::Unknown => gpu_allocator::MemoryLocation::Unknown,
            MemoryUsage::GpuOnly => gpu_allocator::MemoryLocation::GpuOnly,
            MemoryUsage::CpuToGpu => gpu_allocator::MemoryLocation::CpuToGpu,
            MemoryUsage::GpuToCpu => gpu_allocator::MemoryLocation::GpuToCpu,
        }
    }
}

struct InnerDevice {
    window: winit::window::Window,
    _entry: ash::Entry,
    instance: ash::Instance,
    surface_loader: ash::extensions::khr::Surface,
    surface: vk::SurfaceKHR,
    swapchain_loader: ash::extensions::khr::Swapchain,
    debug_utils_loader: ash::extensions::ext::DebugUtils,
    debug_messenger: vk::DebugUtilsMessengerEXT,
    validation_enabled: bool,
    allocator: RefCell<gpu_allocator::vulkan::Allocator>,

    physical_device: vk::PhysicalDevice,
    _memory_properties: vk::PhysicalDeviceMemoryProperties,
    device: ash::Device,
    queue_families: Vec<QueueFamily>,
    super_queue_family: QueueFamily,
}

impl std::fmt::Debug for InnerDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InnerDevice")
            .field("window", &self.window)
            .field("surface", &self.surface)
            .field("physical_device", &self.physical_device)
            .finish()
    }
}

impl InnerDevice {
    pub fn new(event_loop: &EventLoop<()>, builder: DeviceBuilder) -> Result<Rc<Self>> {
        let window_title = &builder.window_title;
        let (window_width, window_height) = builder.window_size;
        let window = window::init_window(
            event_loop,
            window_title,
            window_width,
            window_height,
        );
        let entry = unsafe {
            match ash::Entry::new() {
                Ok(v) => v,
                Err(e) => return Err(Error::external(e, "Failed to load Vulkan library")),
            }
        };
        let instance = utils::create_instance(
            &entry,
            window_title,
            ENGINE_NAME,
            builder.application_version,
            ENGINE_VERSION,
            VULKAN_API_VERSION,
        )?;
        let (debug_utils_loader, debug_messenger) = debug::setup_debug_utils(&entry, &instance);
        let surface = platforms::create_surface(&entry, &instance, &window, &builder.windowing_prefs)?;
        let surface_loader = ash::extensions::khr::Surface::new(&entry, &instance);
        let physical_device = utils::pick_physical_device(
            &instance,
            &surface_loader,
            surface,
            builder.get_extensions(),
        )?;
        let memory_properties = unsafe {
            instance.get_physical_device_memory_properties(physical_device)
        };
        let queue_families = get_queue_families(
            &instance,
            physical_device,
            surface,
            &surface_loader,
        )?;
        let mut super_queue_family = None;
        // This iterator is reversed, and we don't bail out once we find a suitable queue.
        // Why?  Because we want the lowest-indexed queue family that matches, since that
        // will be the best one in most real-world cases, based on what I've seen.
        for qf in queue_families.iter().rev() {
            if qf.can_do_graphics() && qf.can_do_compute() && qf.can_do_transfer() && qf.can_present() {
                super_queue_family = Some(*qf);
            }
        }
        if super_queue_family.is_none() {
            // Maybe the driver is lying about not being able to do transfers.
            for qf in queue_families.iter().rev() {
                if qf.can_do_graphics() && qf.can_do_compute() && qf.can_present() {
                    // The queue can do graphics, compute, and presentation.  Surely it
                    // can also do transfers, right?
                    super_queue_family = Some(*qf);
                }
            }

            if super_queue_family.is_none() {
                // Maybe the driver is lying about not being able to do transfers AND presentation.
                for qf in queue_families.iter().rev() {
                    if qf.can_do_graphics() && qf.can_do_compute() {
                        // The queue can do graphics and compute.  Let's assume it can also
                        // transfer and present and is just lying about its capabilities.
                        super_queue_family = Some(*qf);
                    }
                }
            }
        }
        if super_queue_family.is_none() {
            // Screw you and your funky hardware.
            // TODO: this may need to be handled differently if we want to support headless compute.
            panic!("No super queue found.  What kind of hardware is this?");
        }
        let super_queue_family = super_queue_family.unwrap();
        let device = create_logical_device(
            &instance,
            physical_device,
            &queue_families,
            builder.get_extensions(),
            builder.features_chain,
        );
        let swapchain_loader = ash::extensions::khr::Swapchain::new(&instance, &device);

        let allocator = RefCell::new(utils::create_allocator(
            &instance,
            &physical_device,
            &device,
        )?);

        Ok(Rc::new(Self{
            window,
            _entry: entry,
            instance,
            surface_loader,
            surface,
            swapchain_loader,
            debug_utils_loader,
            debug_messenger,
            validation_enabled: builder.validation_enabled,
            allocator,

            physical_device,
            _memory_properties: memory_properties,
            device: device.clone(),
            queue_families,
            super_queue_family,
        }))
    }

    fn choose_swapchain_extent(
        &self,
        capabilities: &vk::SurfaceCapabilitiesKHR,
    ) -> vk::Extent2D {
        if capabilities.current_extent.width != u32::max_value() {
            capabilities.current_extent
        } else {
            use num::clamp;

            let window_size = self.window.inner_size();

            vk::Extent2D{
                width: clamp(
                    window_size.width,
                    capabilities.min_image_extent.width,
                    capabilities.max_image_extent.width,
                ),
                height: clamp(
                    window_size.height,
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
                ),
            }
        }
    }

    fn create_buffer(
        &self,
        name: &str,
        usage: MemoryUsage,
        buffer_info: &vk::BufferCreateInfo,
    ) -> Result<(vk::Buffer, gpu_allocator::vulkan::Allocation)> {
        use gpu_allocator::vulkan::*;
        let buffer = unsafe {
            Error::wrap_result(
                self.device.create_buffer(&buffer_info, None),
                "Failed to create buffer",
            )?
        };
        let requirements = unsafe {
            self.device.get_buffer_memory_requirements(buffer)
        };
        let allocation_res = self.allocator.borrow_mut().allocate(&AllocationCreateDesc{
            name,
            requirements,
            // I'm going to keep using vk-mem's "usage" terminology.
            // I think it's better.
            location: usage.as_gpu_allocator(),
            linear: true,
        });
        let allocation = match allocation_res {
            Ok(v) => v,
            Err(e) => {
                unsafe {
                    self.device.destroy_buffer(buffer, None);
                }
                return Err(Error::AllocationError(e));
            },
        };
        unsafe {
            let res = self.device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset());
            match res {
                Ok(_) => (),
                Err(e) => match self.allocator.borrow_mut().free(allocation) {
                    Ok(_) => {
                        self.device.destroy_buffer(buffer, None);
                        return Err(Error::wrap(e, "Failed to bind buffer"));
                    },
                    // If we can't free an allocation we just made, things have gone
                    // seriously wrong.  Just bail out.
                    _ => panic!("Failed to free allocation we just made!"),
                },
            }
        }
        Ok((buffer, allocation))
    }

    fn destroy_buffer(
        &self,
        buffer: vk::Buffer,
        allocation: gpu_allocator::vulkan::Allocation,
    ) -> Result<()> {
        match self.allocator.borrow_mut().free(allocation) {
            Ok(_) => (),
            Err(e) => return Err(Error::AllocationError(e)),
        };
        unsafe {
            self.device.destroy_buffer(buffer, None);
        }
        Ok(())
    }

    fn create_image(
        &self,
        name: &str,
        usage: MemoryUsage,
        image_info: &vk::ImageCreateInfo,
    ) -> Result<(vk::Image, gpu_allocator::vulkan::Allocation)> {
        use gpu_allocator::vulkan::*;
        let image = unsafe {
            Error::wrap_result(
                self.device.create_image(&image_info, None),
                "Failed to create image",
            )?
        };
        let requirements = unsafe {
            self.device.get_image_memory_requirements(image)
        };
        let allocation_res = self.allocator.borrow_mut().allocate(&AllocationCreateDesc{
            name,
            requirements,
            // I'm going to keep using vk-mem's "usage" terminology.
            // I think it's better.
            location: usage.as_gpu_allocator(),
            linear: image_info.tiling == vk::ImageTiling::LINEAR,
        });
        let allocation = match allocation_res {
            Ok(v) => v,
            Err(e) => {
                unsafe {
                    self.device.destroy_image(image, None);
                }
                return Err(Error::AllocationError(e));
            },
        };
        unsafe {
            let res = self.device.bind_image_memory(image, allocation.memory(), allocation.offset());
            match res {
                Ok(_) => (),
                Err(e) => match self.allocator.borrow_mut().free(allocation) {
                    Ok(_) => {
                        self.device.destroy_image(image, None);
                        return Err(Error::wrap(e, "Failed to bind image"));
                    },
                    // If we can't free an allocation we just made, things have gone
                    // seriously wrong.  Just bail out.
                    _ => panic!("Failed to free allocation we just made!"),
                },
            }
        }
        Ok((image, allocation))
    }

    fn destroy_image(
        &self,
        image: vk::Image,
        allocation: gpu_allocator::vulkan::Allocation,
    ) -> Result<()> {
        match self.allocator.borrow_mut().free(allocation) {
            Ok(_) => (),
            Err(e) => return Err(Error::AllocationError(e)),
        };
        unsafe {
            self.device.destroy_image(image, None);
        }
        Ok(())
    }

    fn query_swapchain_support(&self) -> Result<utils::SwapChainSupport> {
        utils::query_swapchain_support(
            self.physical_device,
            &self.surface_loader,
            self.surface,
        )
    }

    fn create_swapchain(&self, swapchain_create_info: &vk::SwapchainCreateInfoKHR) -> Result<vk::SwapchainKHR> {
        Ok(unsafe {
            Error::wrap_result(
                self.swapchain_loader
                    .create_swapchain(swapchain_create_info, None),
                "Failed to create swapchain",
            )?
        })
    }

    fn get_swapchain_images(&self, swapchain: vk::SwapchainKHR) -> Result<Vec<vk::Image>> {
        Ok(unsafe {
            Error::wrap_result(
                self.swapchain_loader
                    .get_swapchain_images(swapchain),
                "Failed to get swapchain images",
            )?
        })
    }

    fn destroy_swapchain(&self, swapchain: vk::SwapchainKHR) {
        unsafe {
            self.swapchain_loader.destroy_swapchain(swapchain, None);
        }
    }

    fn acquire_next_image(
        &self,
        swapchain: vk::SwapchainKHR,
        timeout: u64,
        semaphore: vk::Semaphore,
        fence: vk::Fence
    ) -> ash::prelude::VkResult<(u32, bool)> {
        unsafe {
            self.swapchain_loader
                .acquire_next_image(
                    swapchain,
                    timeout,
                    semaphore,
                    fence,
                )
        }
    }

    fn queue_present(
        &self,
        queue: &Queue,
        present_info: &vk::PresentInfoKHR
    ) -> ash::prelude::VkResult<bool> {
        unsafe {
            self.swapchain_loader
                .queue_present(queue.get(), present_info)
        }
    }
}

impl Drop for InnerDevice {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);

            if self.validation_enabled {
                self.debug_utils_loader
                    .destroy_debug_utils_messenger(self.debug_messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

fn create_logical_device(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    queue_familes: &Vec<QueueFamily>,
    enabled_extensions: &[String],
    features_chain: *mut c_void,
) -> ash::Device {
    let device_properties = unsafe{ instance.get_physical_device_properties(physical_device) };
    let device_name = vk_to_string(&device_properties.device_name);
    // This is for a temporary workaround.
    let is_intel = device_name.contains("Intel");

    let mut queue_create_infos = vec![];
    // This needs to be outside the loop to avoid use-after-free problems with the pointer
    // stuff going on in DeviceQueueCreateInfo below.
    let mut priority = 1.0_f32;
    let mut queue_priorities = vec![];
    // TODO: I hate this.  Queue priorities and probably also queue creation
    // should be in the user's hands (albeit with a "do it for me" option
    // that sets things up in a sensible manner).  I don't really have the
    // time or the inclination to fix it right now, though.
    for queue_family in queue_familes.iter() {
        for _ in queue_priorities.len()..queue_family.num_queues as usize {
            queue_priorities.push(priority);
            priority = priority / 2_f32;
        }
        queue_create_infos.push(vk::DeviceQueueCreateInfo{
            s_type: vk::StructureType::DEVICE_QUEUE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::DeviceQueueCreateFlags::empty(),
            queue_family_index: queue_family.idx,
            p_queue_priorities: queue_priorities.as_ptr(),
            queue_count: queue_family.num_queues,
        });
    }

    let mut physical_device_features = vk::PhysicalDeviceFeatures{
        ..Default::default()
    };
    physical_device_features.independent_blend = vk::TRUE;
    // https://gitlab.freedesktop.org/mesa/mesa/-/merge_requests/7329
    // This is a workaround until the above change hits Ubuntu repos.
    // The funny thing is that as far as I can tell, shaders that use
    // uint64_t work just fine without it, but the validation layer complains.
    if !is_intel {
        physical_device_features.shader_int64 = vk::TRUE;
    }

    let mut device_address_features = vk::PhysicalDeviceBufferDeviceAddressFeatures{
        s_type: vk::StructureType::PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES,
        p_next: features_chain,
        buffer_device_address: vk::TRUE,
        ..Default::default()
    };

    let mut imageless_framebuffer_features = vk::PhysicalDeviceImagelessFramebufferFeatures{
        s_type: vk::StructureType::PHYSICAL_DEVICE_IMAGELESS_FRAMEBUFFER_FEATURES,
        p_next: (&mut device_address_features as *mut _) as *mut c_void,
        imageless_framebuffer: vk::TRUE,
        ..Default::default()
    };
    //descriptor_indexing_features.shader_uniform_buffer_array_non_uniform_indexing = vk::TRUE;

    //physical_device_features.shader_uniform_buffer_array_dynamic_indexing = vk::TRUE;
    physical_device_features.shader_sampled_image_array_dynamic_indexing = vk::TRUE;
    physical_device_features.shader_storage_buffer_array_dynamic_indexing = vk::TRUE;
    physical_device_features.shader_storage_image_array_dynamic_indexing = vk::TRUE;
    physical_device_features.sampler_anisotropy = vk::TRUE;

    let physical_device_features2 = vk::PhysicalDeviceFeatures2{
        s_type: vk::StructureType::PHYSICAL_DEVICE_FEATURES_2,
        p_next: (&mut imageless_framebuffer_features as *mut _) as *mut _,
        features: physical_device_features,
    };

    let required_validation_layer_raw_names: Vec<CString> = VALIDATION
        .required_validation_layers
        .iter()
        .map(|layer_name| CString::new(*layer_name).unwrap())
        .collect();
    let enable_layer_names: Vec<*const c_char> = required_validation_layer_raw_names
        .iter()
        .map(|layer_name| layer_name.as_ptr())
        .collect();

    let mut enable_extension_names = vec![
        ash::extensions::khr::Swapchain::name().as_ptr(),
    ];
    // All this crap because a vk struct INSISTS on using *const c_char!
    let mut extension_names_list = vec![];
    for ext in enabled_extensions.iter() {
        extension_names_list.push(CString::new(ext.as_str()).unwrap());
    }
    for ext in extension_names_list.iter() {
        enable_extension_names.push(ext.as_ptr());
    }

    let p_next: *const c_void = &physical_device_features2 as *const _ as *const _;

    let device_create_info = vk::DeviceCreateInfo{
        s_type: vk::StructureType::DEVICE_CREATE_INFO,
        p_next,
        flags: vk::DeviceCreateFlags::empty(),
        queue_create_info_count: queue_create_infos.len() as u32,
        p_queue_create_infos: queue_create_infos.as_ptr(),
        enabled_layer_count: if VALIDATION.is_enabled {
            enable_layer_names.len()
        } else {
            0
        } as u32,
        pp_enabled_layer_names: if VALIDATION.is_enabled {
            enable_layer_names.as_ptr()
        } else {
            ptr::null()
        },
        enabled_extension_count: enable_extension_names.len() as u32,
        pp_enabled_extension_names: enable_extension_names.as_ptr(),
        p_enabled_features: ptr::null(),
    };

    let device: ash::Device = unsafe {
        instance
            .create_device(physical_device, &device_create_info, None)
            .expect("Failed to create logical device!")
    };

    device
}

#[derive(Copy, Clone, Debug)]
pub struct QueueFamilyRef {
    idx: u32,
}

impl QueueFamilyRef {
    pub fn matches(&self, queue: &Queue) -> bool {
        self.idx == queue.family_idx
    }
}

impl std::fmt::Display for QueueFamilyRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.idx.fmt(f)
    }
}

#[derive(Copy, Clone)]
pub struct QueueFamily {
    idx: u32,
    flags: vk::QueueFlags,
    num_queues: u32,
    can_present: bool,
}

impl QueueFamily {
    pub fn get_ref(&self) -> QueueFamilyRef {
        QueueFamilyRef{
            idx: self.idx,
        }
    }

    pub fn get_queue(&self, device: &Device, index: u32) -> Result<Queue> {
        if index >= self.num_queues {
            return Err(Error::InvalidQueueIndex(index, self.idx));
        }
        Queue::new(
            Rc::clone(&device.inner),
            self.idx,
            index,
            self.flags,
            self.can_present,
        )
    }

    pub fn can_do_graphics(&self) -> bool {
        self.flags.contains(vk::QueueFlags::GRAPHICS)
    }

    pub fn can_present(&self) -> bool {
        self.can_present
    }

    pub fn can_do_compute(&self) -> bool {
        self.flags.contains(vk::QueueFlags::COMPUTE)
    }

    pub fn can_do_transfer(&self) -> bool {
        self.flags.contains(vk::QueueFlags::TRANSFER)
    }

    pub fn can_do_sparse_binding(&self) -> bool {
        self.flags.contains(vk::QueueFlags::SPARSE_BINDING)
    }
}

fn get_queue_families(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    surface_loader: &ash::extensions::khr::Surface,
) -> Result<Vec<QueueFamily>> {
    let queue_families = unsafe {
        instance.get_physical_device_queue_family_properties(physical_device)
    };

    let mut infos = vec![];

    for (i, queue_family) in queue_families.iter().enumerate() {
        let mut queues = vec![];
        for j in 0..queue_family.queue_count {
            queues.push(j as u32);
        }
        infos.push(QueueFamily{
            idx: i as u32,
            flags: queue_family.queue_flags,
            num_queues: queue_family.queue_count,
            can_present: unsafe {
                Error::wrap_result(
                    surface_loader.get_physical_device_surface_support(
                        physical_device,
                        i as u32,
                        surface,
                    ),
                    "Failed to get physical device surface support info (needed to determine presentation support)",
                )?
            },
        });
    }

    Ok(infos)
}

pub trait NamedResource {
    fn name(&self) -> &str;
}

pub trait HasHandle {
    fn vk_handle(&self) -> u64;
}
