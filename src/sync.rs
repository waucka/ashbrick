//! This module exposes the Vulkan synchronization objects in
//! an idiomatic way.

use ash::vk;

use std::ptr;
use std::rc::Rc;

use super::{Device, InnerDevice};

use super::errors::{Error, Result};

pub struct Semaphore {
    device: Rc<InnerDevice>,
    pub (crate) semaphore: vk::Semaphore,
    name: String,
}

impl Semaphore {
    pub fn new(device: &Device, name: &str) -> Result<Self> {
        let semaphore_create_info = vk::SemaphoreCreateInfo{
            s_type: vk::StructureType::SEMAPHORE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::SemaphoreCreateFlags::empty(),
        };
        Ok(Self{
            device: Rc::clone(&device.inner),
            semaphore: unsafe {
                Error::wrap_result(
                    device.inner.device
                        .create_semaphore(&semaphore_create_info, None),
                    "Failed to create semaphore",
                )?
            },
            name: String::from(name),
        })
    }
}

impl Drop for Semaphore {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_semaphore(self.semaphore, None);
        }
    }
}

impl super::NamedResource for Semaphore {
    fn name(&self) -> &str {
        &self.name
    }
}

pub struct Fence {
    device: Rc<InnerDevice>,
    pub (crate) fence: vk::Fence,
    name: String,
}

impl Fence {
    pub fn new(device: &Device, name: &str, signaled: bool) -> Result<Self> {
        Self::new_internal(&device.inner, name, signaled)
    }

    pub (crate) fn new_internal(device: &Rc<InnerDevice>, name: &str, signaled: bool) -> Result<Self> {
        let fence_create_info = vk::FenceCreateInfo{
            s_type: vk::StructureType::FENCE_CREATE_INFO,
            p_next: ptr::null(),
            flags: if signaled {
                vk::FenceCreateFlags::SIGNALED
            } else {
                vk::FenceCreateFlags::empty()
            },
        };
        Ok(Self{
            device: Rc::clone(&device),
            fence: unsafe {
                Error::wrap_result(
                    device.device
                        .create_fence(&fence_create_info, None),
                    "Failed to create command buffer fence",
                )?
            },
            name: String::from(name),
        })
    }

    pub fn wait(&self, timeout: u64) -> Result<()> {
        let wait_fences = [self.fence];
        unsafe {
            Error::wrap_result(
                self.device.device.wait_for_fences(
                    &wait_fences,
                    true,
                    timeout,
                ),
                "Failed to wait for fence",
            )
        }
    }

    pub fn reset(&self) -> Result<()> {
        let reset_fences = [self.fence];
        unsafe {
            Error::wrap_result(
                self.device.device.reset_fences(
                    &reset_fences,
                ),
                "Failed to wait for fence",
            )
        }
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_fence(self.fence, None);
        }
    }
}

impl super::NamedResource for Fence {
    fn name(&self) -> &str {
        &self.name
    }
}
