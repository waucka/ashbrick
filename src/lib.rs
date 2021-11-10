use ash::vk;
use winit::event_loop::EventLoop;

pub use ash;
pub use winit;
pub use crevice;
#[cfg(feature = "egui")]
pub use egui;

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
pub mod image;
pub mod renderer;
pub mod shader;
pub mod texture;
pub mod descriptor;

use command_buffer::CommandPool;

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

    pub fn can_do_graphics(&self) -> bool {
        self.flags.contains(vk::QueueFlags::GRAPHICS)
    }

    pub fn can_present(&self) -> bool {
        self.can_present
    }

    #[allow(unused)]
    pub fn can_do_compute(&self) -> bool {
        self.flags.contains(vk::QueueFlags::COMPUTE)
    }

    pub fn can_do_transfer(&self) -> bool {
        self.flags.contains(vk::QueueFlags::TRANSFER)
    }

    #[allow(unused)]
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

    #[allow(unused)]
    pub fn with_extension(mut self, extension_name: &str) -> Self {
        self.extensions.push(extension_name.to_string());
        self
    }

    pub fn with_windowing_prefs(mut self, prefs: platforms::WindowingPreferences) -> Self {
        self.windowing_prefs = prefs;
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

    #[allow(unused)]
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

    pub fn get_default_graphics_queue(&self) -> Rc<Queue> {
        self.inner.get_default_graphics_queue()
    }

    pub fn get_default_graphics_pool(&self) -> Rc<CommandPool> {
        self.inner.get_default_graphics_pool()
    }

    #[allow(unused)]
    pub fn get_default_present_queue(&self) -> Rc<Queue> {
        self.inner.get_default_present_queue()
    }

    pub fn get_default_transfer_queue(&self) -> Rc<Queue> {
        self.inner.get_default_transfer_queue()
    }

    pub fn get_default_transfer_pool(&self) -> Rc<CommandPool> {
        self.inner.get_default_transfer_pool()
    }

    #[allow(unused)]
    pub fn get_queues(&self) -> Vec<Rc<Queue>> {
        let mut queues = vec![];
        for q in &self.inner.queue_set.borrow().queues {
            queues.push(q.clone());
        }
        queues
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
}

pub enum MemoryUsage {
    #[allow(dead_code)]
    Unknown,
    #[allow(dead_code)]
    GpuOnly,
    #[allow(dead_code)]
    CpuToGpu,
    #[allow(dead_code)]
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

struct QueueSet {
    queues: Vec<Rc<Queue>>,
    pools: Vec<Rc<CommandPool>>,
    // These three are indexes into the above vector ("queues").
    default_graphics_queue_idx: usize,
    default_present_queue_idx: usize,
    default_transfer_queue_idx: usize,
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
    queue_set: RefCell<QueueSet>,
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
        let queue_infos = get_queue_info(
            &instance,
            physical_device,
            surface,
            &surface_loader,
        )?;
        let device = create_logical_device(
            &instance,
            physical_device,
            &queue_infos,
            builder.get_extensions(),
        );
        let swapchain_loader = ash::extensions::khr::Swapchain::new(&instance, &device);

        let allocator = RefCell::new(utils::create_allocator(
            &instance,
            &physical_device,
            &device,
        )?);

        let this = Rc::new(Self{
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
            queue_set: RefCell::new(QueueSet {
                queues: Vec::new(),
                pools: Vec::new(),
                default_graphics_queue_idx: 0,
                default_present_queue_idx: 0,
                default_transfer_queue_idx: 0,
            }),
        });

        {
            let queues = get_queues_from_device(
                this.clone(),
                queue_infos,
            )?;
            let mut queue_set = this.queue_set.borrow_mut();

            let mut maybe_graphics_queue_idx = None;
            let mut maybe_present_queue_idx = None;
            let mut maybe_transfer_queue_idx = None;
            for (idx, queue) in queues.iter().enumerate() {
                if queue.can_do_graphics() {
                    maybe_graphics_queue_idx = Some(idx)
                }
                if queue.can_present() {
                    maybe_present_queue_idx = Some(idx)
                }
                if queue.can_do_transfer() {
                    maybe_transfer_queue_idx = Some(idx)
                }
            }

            let (default_graphics_queue_idx, default_present_queue_idx, default_transfer_queue_idx) =
                match (maybe_graphics_queue_idx, maybe_present_queue_idx, maybe_transfer_queue_idx) {
                    (Some(q1), Some(q2), Some(q3)) => (q1, q2, q3),
                    _ => panic!("Unable to create all three of: graphics queue, present queue, transfer queue!"),
                };

            let mut pools = Vec::new();
            for q in queues.iter() {
                pools.push(CommandPool::from_inner(
                    Rc::clone(&this),
                    Rc::clone(q),
                    false,
                    false,
                )?);
            }

            queue_set.queues = queues;
            queue_set.pools = pools;
            queue_set.default_graphics_queue_idx = default_graphics_queue_idx;
            queue_set.default_present_queue_idx = default_present_queue_idx;
            queue_set.default_transfer_queue_idx = default_transfer_queue_idx;
        }

        Ok(this)
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

    fn get_default_graphics_queue(&self) -> Rc<Queue> {
        let queue_set = self.queue_set.borrow();
        Rc::clone(&queue_set.queues[queue_set.default_graphics_queue_idx])
    }

    fn get_default_graphics_pool(&self) -> Rc<CommandPool> {
        let queue_set = self.queue_set.borrow();
        Rc::clone(&queue_set.pools[queue_set.default_graphics_queue_idx])
    }

    #[allow(unused)]
    fn get_default_present_queue(&self) -> Rc<Queue> {
        let queue_set = self.queue_set.borrow();
        Rc::clone(&queue_set.queues[queue_set.default_present_queue_idx])
    }

    fn get_default_transfer_queue(&self) -> Rc<Queue> {
        let queue_set = self.queue_set.borrow();
        Rc::clone(&queue_set.queues[queue_set.default_transfer_queue_idx])
    }

    fn get_default_transfer_pool(&self) -> Rc<CommandPool> {
        let queue_set = self.queue_set.borrow();
        Rc::clone(&queue_set.pools[queue_set.default_transfer_queue_idx])
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
        queue: Rc<Queue>,
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
            if let Ok(mut queue_set) = self.queue_set.try_borrow_mut() {
                for q in queue_set.queues.drain(..) {
                    if Rc::strong_count(&q) > 1 {
                        panic!("We are destroying a Device, but a queue is still in use!");
                    }
                }
            } else {
                panic!("We are destroying a Device, but its queue set is borrowed!");
            }
            // I don't think I have to destroy the allocator before the device; its
            // Drop implementation only seems to log some debug info.
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
    queue_infos: &Vec<QueueInfo>,
    enabled_extensions: &[String],
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
    // This will fail horribly if the queue IDs are not consecutive.
    // Since the Vulkan API assumes they are, I don't think there are
    // any plausible cases where they won't be.
    for queue_info in queue_infos.iter() {
        for _ in queue_priorities.len()..queue_info.queues.len() {
            queue_priorities.push(priority);
            priority = priority / 2_f32;
        }
        queue_create_infos.push(vk::DeviceQueueCreateInfo{
            s_type: vk::StructureType::DEVICE_QUEUE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::DeviceQueueCreateFlags::empty(),
            queue_family_index: queue_info.family_idx,
            p_queue_priorities: queue_priorities.as_ptr(),
            queue_count: queue_info.queues.len() as u32,
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

    let mut imageless_framebuffer_features = vk::PhysicalDeviceImagelessFramebufferFeatures{
        s_type: vk::StructureType::PHYSICAL_DEVICE_IMAGELESS_FRAMEBUFFER_FEATURES,
        p_next: ptr::null_mut(),
        ..Default::default()
    };
    imageless_framebuffer_features.imageless_framebuffer = vk::TRUE;

    let descriptor_indexing_features = {
        let mut descriptor_indexing_features = vk::PhysicalDeviceDescriptorIndexingFeatures{
            s_type: vk::StructureType::PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES,
            p_next: (&mut imageless_framebuffer_features as *mut _) as *mut c_void,
            ..Default::default()
        };
        descriptor_indexing_features.runtime_descriptor_array = vk::TRUE;
        //descriptor_indexing_features.shader_uniform_buffer_array_non_uniform_indexing = vk::TRUE;
        descriptor_indexing_features.shader_sampled_image_array_non_uniform_indexing = vk::TRUE;
        descriptor_indexing_features.shader_storage_buffer_array_non_uniform_indexing = vk::TRUE;
        descriptor_indexing_features.shader_storage_image_array_non_uniform_indexing = vk::TRUE;
        descriptor_indexing_features.descriptor_binding_partially_bound = vk::TRUE;
        descriptor_indexing_features
    };

    //physical_device_features.shader_uniform_buffer_array_dynamic_indexing = vk::TRUE;
    physical_device_features.shader_sampled_image_array_dynamic_indexing = vk::TRUE;
    physical_device_features.shader_storage_buffer_array_dynamic_indexing = vk::TRUE;
    physical_device_features.shader_storage_image_array_dynamic_indexing = vk::TRUE;
    physical_device_features.sampler_anisotropy = vk::TRUE;

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

    let p_next: *const c_void = &descriptor_indexing_features as *const _ as *const _;

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
        p_enabled_features: &physical_device_features,
    };

    let device: ash::Device = unsafe {
        instance
            .create_device(physical_device, &device_create_info, None)
            .expect("Failed to create logical device!")
    };

    device
}

struct QueueInfo {
    family_idx: u32,
    flags: vk::QueueFlags,
    queues: Vec<u32>,
    can_present: bool,
}

fn get_queue_info(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    surface_loader: &ash::extensions::khr::Surface,
) -> Result<Vec<QueueInfo>> {
    let queue_families = unsafe {
        instance.get_physical_device_queue_family_properties(physical_device)
    };

    let mut infos = vec![];

    for (i, queue_family) in queue_families.iter().enumerate() {
        let mut queues = vec![];
        for j in 0..queue_family.queue_count {
            queues.push(j as u32);
        }
        infos.push(QueueInfo{
            family_idx: i as u32,
            flags: queue_family.queue_flags,
            queues: queues,
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

fn get_queues_from_device(
    device: Rc<InnerDevice>,
    queue_infos: Vec<QueueInfo>
) -> Result<Vec<Rc<Queue>>> {
    let mut queues = vec![];
    for queue_info in queue_infos.iter() {
        if queue_info.queues.len() == 0 {
            println!("A queue family with no queues in it?  This driver is on crack!");
            continue;
        }

        for queue_idx in queue_info.queues.iter() {
            queues.push(Rc::new(Queue::new(
                device.clone(),
                queue_info.family_idx,
                *queue_idx,
                queue_info.flags,
                queue_info.can_present,
            )?));
        }
    }

    Ok(queues)
}

pub trait GraphicsResource {}
