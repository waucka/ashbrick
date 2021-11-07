use ash::vk;
use glsl_layout::Uniform;

use std::rc::Rc;
use std::marker::PhantomData;
use std::ptr;

use super::errors::{Error, Result};

use super::{Device, InnerDevice, MemoryUsage};
use super::command_buffer::{CommandBuffer, CommandPool};

pub trait HasBuffer {
    fn get_buffer(&self) -> vk::Buffer;
}

// With gpu-allocator instead of vk-mem, I don't think this type
// should even exist.
pub struct MemoryMapping<T> {
    allocation: gpu_allocator::vulkan::Allocation,
    _phantom: PhantomData<T>,
}

impl<T> MemoryMapping<T> {
    fn new(buf: &Buffer) -> Result<Self> {
        Ok(Self {
            allocation: buf.allocation.clone(),
            _phantom: PhantomData,
        })
    }

    fn copy_slice(&self, src: &[T]) -> Result<()> {
        unsafe {
            let data_ptr = match self.allocation.mapped_ptr() {
                Some(v) => v.as_ptr() as *mut T,
                None => return Err(Error::UnmappableBuffer),
            };
            data_ptr.copy_from_nonoverlapping(src.as_ptr(), src.len());
        }
        Ok(())
    }

    fn copy_item(&self, src: &T) -> Result<()> {
        unsafe {
            let data_ptr = match self.allocation.mapped_ptr() {
                Some(v) => v.as_ptr() as *mut T,
                None => return Err(Error::UnmappableBuffer),
            };
            data_ptr.copy_from_nonoverlapping(src, 1);
        }
        Ok(())
    }
}

pub struct Buffer {
    device: Rc<InnerDevice>,
    pub (in super) buf: vk::Buffer,
    allocation: gpu_allocator::vulkan::Allocation,
    size: vk::DeviceSize,
}

impl super::GraphicsResource for Buffer {}

impl Buffer {
    fn new(
        device: Rc<InnerDevice>,
        name: &str,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        memory_usage: MemoryUsage,
        sharing_mode: vk::SharingMode,
    ) -> Result<Self> {
        if size == 0 {
            return Err(Error::ZeroSizeBuffer);
        }
        let buffer_create_info = vk::BufferCreateInfo{
            s_type: vk::StructureType::BUFFER_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::BufferCreateFlags::empty(),
            size: size,
            usage: usage,
            sharing_mode: sharing_mode,
            queue_family_index_count: 0,
            p_queue_family_indices: ptr::null(),
        };

        let (buffer, allocation) = device.create_buffer(
            name,
            memory_usage,
            &buffer_create_info,
        )?;

        Ok(Buffer{
            device: device.clone(),
            buf: buffer,
            allocation,
            size,
        })
    }

    pub fn copy(
        src_buffer: Rc<Buffer>,
        dst_buffer: Rc<Buffer>,
        pool: Rc<CommandPool>,
    ) -> Result<()> {
        if src_buffer.size > dst_buffer.size {
            return Err(Error::InvalidBufferCopy(src_buffer.size, dst_buffer.size));
        }
        CommandBuffer::run_oneshot_internal(
           src_buffer.device.clone(),
            pool,
            |writer| {
                let copy_regions = [vk::BufferCopy{
                    src_offset: 0,
                    dst_offset: 0,
                    size: src_buffer.size,
                }];

                writer.copy_buffer(
                    Rc::clone(&src_buffer),
                    Rc::clone(&dst_buffer),
                    &copy_regions,
                );
                Ok(())
            }
        )
    }

    pub fn with_memory_mapping<T, F>(&self, mmap_fn: F) -> Result<()>
    where
        F: Fn(&MemoryMapping<T>) -> Result<()> {
        let mmap = MemoryMapping::new(self)?;
        mmap_fn(&mmap)
    }
}

impl HasBuffer for Buffer {
    fn get_buffer(&self) -> vk::Buffer {
        self.buf
    }
}


impl Drop for Buffer {
    fn drop(&mut self) {
        match self.device.destroy_buffer(self.buf, self.allocation.clone()) {
            Ok(_) => (),
            Err(e) => println!("Failed to destroy buffer: {}", e),
        }
    }
}

pub struct UploadSourceBuffer {
    buf: Rc<Buffer>,
}

impl UploadSourceBuffer {
    pub fn new(
        device: &Device,
        name: &str,
        size: vk::DeviceSize,
    ) -> Result<Self> {
        Ok(Self {
            buf: Rc::new(Buffer::new(
                Rc::clone(&device.inner),
                name,
                size,
                vk::BufferUsageFlags::TRANSFER_SRC,
                MemoryUsage::CpuToGpu,
                vk::SharingMode::EXCLUSIVE,
            )?),
        })
    }

    pub fn copy_data<T>(&self, data: &[T]) -> Result<()> {
        self.buf.with_memory_mapping(|mmap| {
            mmap.copy_slice(data)?;
            Ok(())
        })
    }
}

impl HasBuffer for UploadSourceBuffer {
    fn get_buffer(&self) -> vk::Buffer {
        self.buf.buf
    }
}

impl Drop for UploadSourceBuffer {
    fn drop(&mut self) {
        // Drop has been implemented solely so that UploadSourceBuffers can be recorded as
        // dependencies for CommandBuffers.
    }
}

impl super::GraphicsResource for UploadSourceBuffer {}

pub struct VertexBuffer<T> {
    buf: Rc<Buffer>,
    len: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> super::GraphicsResource for VertexBuffer<T> {}

impl<T> VertexBuffer<T> {
    pub fn new(
        device: &Device,
        name: &str,
        data: &[T],
    ) -> Result<Self> {
        let buffer_size = std::mem::size_of_val(data) as vk::DeviceSize;
        let upload_buffer = UploadSourceBuffer::new(device, "temp-upload-source-buffer", buffer_size)?;
        upload_buffer.copy_data(data)?;
        let vertex_buffer = Rc::new(Buffer::new(
            Rc::clone(&device.inner),
            name,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
            MemoryUsage::GpuOnly,
            // TODO: is this really what we want?
            vk::SharingMode::EXCLUSIVE,
        )?);

        Buffer::copy(
            Rc::clone(&upload_buffer.buf),
            Rc::clone(&vertex_buffer),
            device.inner.get_default_transfer_pool(),
        )?;

        Ok(Self{
            buf: vertex_buffer,
            len: data.len(),
            _phantom: std::marker::PhantomData,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }
}

impl<V> HasBuffer for VertexBuffer<V> {
    fn get_buffer(&self) -> vk::Buffer {
        self.buf.buf
    }
}

impl<V> Drop for VertexBuffer<V> {
    fn drop(&mut self) {
        // Drop has been implemented solely so that VertexBuffers can be recorded as
        // dependencies for CommandBuffers.
    }
}

pub struct IndexBuffer {
    buf: Rc<Buffer>,
    len: usize,
}

impl super::GraphicsResource for IndexBuffer {}

impl IndexBuffer {
    #[allow(unused)]
    pub fn new(
        device: &Device,
        name: &str,
        data: &[u32],
    ) -> Result<Self> {
        let buffer_size = std::mem::size_of_val(data) as vk::DeviceSize;
        let upload_buffer = UploadSourceBuffer::new(device, "temp-upload-source-buffer", buffer_size)?;
        upload_buffer.copy_data(data)?;
        let index_buffer = Rc::new(Buffer::new(
            Rc::clone(&device.inner),
            name,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
            MemoryUsage::GpuOnly,
            // TODO: Is this actually what we want in multiqueue scenarios?
            vk::SharingMode::EXCLUSIVE,
        )?);

        Buffer::copy(
            Rc::clone(&upload_buffer.buf),
            Rc::clone(&index_buffer),
            device.inner.get_default_transfer_pool(),
        )?;

        Ok(Self{
            buf: index_buffer,
            len: data.len(),
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }
}

impl HasBuffer for IndexBuffer {
    fn get_buffer(&self) -> vk::Buffer {
        self.buf.buf
    }
}

impl Drop for IndexBuffer {
    fn drop(&mut self) {
        // Drop has been implemented solely so that IndexBuffers can be recorded as
        // dependencies for CommandBuffers.
    }
}

pub struct UniformBuffer<T: Uniform>
where <T as Uniform>::Std140: Sized
{
    buf: Rc<Buffer>,
    size: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Uniform> UniformBuffer<T>
where <T as Uniform>::Std140: Sized
{
    pub fn new(
        device: &Device,
        name: &str,
        initial_value: Option<&T>,
    ) -> Result<Self> {
        let size = std::mem::size_of::<T::Std140>();
        let buffer = Rc::new(Buffer::new(
            Rc::clone(&device.inner),
            name,
            size as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            // TODO: allow GpuOnly for infrequently-updated buffers
            MemoryUsage::CpuToGpu,
            vk::SharingMode::EXCLUSIVE,
        )?);

        let this = Self{
            buf: buffer,
            size,
            _phantom: std::marker::PhantomData,
        };

        if let Some(initval) = initial_value {
            this.update(initval)?;
        }

        Ok(this)
    }

    pub fn update(
        &self,
        new_value: &T,
    ) -> Result<()> {
        let std140_val = new_value.std140();
        let std140_value_size = std::mem::size_of_val::<T::Std140>(&std140_val);
        if self.size < std140_value_size {
            Err(Error::InvalidUniformWrite(std140_value_size, self.size))
        } else {
            self.buf.with_memory_mapping(|mmap| {
                mmap.copy_item(&std140_val)?;
                Ok(())
            })
        }
    }

    pub fn len(&self) -> vk::DeviceSize {
        self.size as vk::DeviceSize
    }
}

impl<T: Uniform> HasBuffer for UniformBuffer<T>
where <T as Uniform>::Std140: Sized
{
    fn get_buffer(&self) -> vk::Buffer {
        self.buf.buf
    }
}

impl<T: Uniform> Drop for UniformBuffer<T>
where <T as Uniform>::Std140: Sized
{
    fn drop(&mut self) {
        // Drop has been implemented solely so that UniformBuffers can be recorded as
        // dependencies for CommandBuffers.
    }
}
