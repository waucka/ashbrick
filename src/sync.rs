use ash::vk;

use std::ptr;
use std::rc::Rc;

use super::{Device, InnerDevice};

use super::errors::{Error, Result};

pub struct Semaphore {
    device: Rc<InnerDevice>,
    pub (crate) semaphore: vk::Semaphore,
}

impl Semaphore {
    pub fn new(device: &Device) -> Result<Self> {
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

pub struct Fence {
    device: Rc<InnerDevice>,
    pub (crate) fence: vk::Fence,
}

impl Fence {
    pub fn new(device: &Device, signaled: bool) -> Result<Self> {
        Self::new_internal(&device.inner, signaled)
    }

    pub (crate) fn new_internal(device: &Rc<InnerDevice>, signaled: bool) -> Result<Self> {
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
