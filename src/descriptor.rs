use ash::vk;
use crevice::std140::{AsStd140, WriteStd140};

use std::cell::RefCell;
use std::collections::HashMap;
use std::os::raw::c_void;
use std::pin::Pin;
use std::ptr;
use std::rc::Rc;

use super::{Device, InnerDevice};
use super::buffer::{UniformBuffer, ComplexUniformBuffer, HasBuffer};
use super::texture::{Texture, Sampler};

use super::errors::{Error, Result};

const DEBUG_DESCRIPTOR_SETS: bool = false;

pub struct WriteDescriptorSet {
    descriptor_type: vk::DescriptorType,
    buffer_info: Pin<Box<Vec<vk::DescriptorBufferInfo>>>,
    image_info: Pin<Box<Vec<vk::DescriptorImageInfo>>>,
    dst_set: Rc<DescriptorSet>,
    dst_binding: u32,
}

impl WriteDescriptorSet {
    fn for_buffers(
        descriptor_type: vk::DescriptorType,
        buffer_info: Vec<vk::DescriptorBufferInfo>,
        dst_set: Rc<DescriptorSet>,
        dst_binding: u32,
    ) -> Self {
        Self{
            descriptor_type,
            buffer_info: Pin::new(Box::new(buffer_info)),
            image_info: Pin::new(Box::new(Vec::new())),
            dst_set,
            dst_binding,
        }
    }

    fn for_images(
        descriptor_type: vk::DescriptorType,
        image_info: Vec<vk::DescriptorImageInfo>,
        dst_set: Rc<DescriptorSet>,
        dst_binding: u32,
    ) -> Self {
        Self{
            descriptor_type,
            buffer_info: Pin::new(Box::new(Vec::new())),
            image_info: Pin::new(Box::new(image_info)),
            dst_set,
            dst_binding,
        }
    }

    fn get(&self) -> vk::WriteDescriptorSet {
        let mut set = vk::WriteDescriptorSet{
            s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
            p_next: ptr::null(),
            dst_set: self.dst_set.inner,
            dst_binding: self.dst_binding,
            dst_array_element: 0,
            descriptor_count: 0,
            descriptor_type: self.descriptor_type,
            p_image_info: ptr::null(),
            p_buffer_info: ptr::null(),
            p_texel_buffer_view: ptr::null(),
        };

        if self.buffer_info.len() > 0 {
            set.descriptor_count = self.buffer_info.len() as u32;
            set.p_buffer_info = self.buffer_info.as_ptr();
        } else if self.image_info.len() > 0 {
            set.descriptor_count = self.image_info.len() as u32;
            set.p_image_info = self.image_info.as_ptr();
        } else {
            panic!("Coding error!");
        }
        if set.descriptor_count == 0 {
            panic!("No descriptors set!");
        }

        set
    }
}

pub trait DescriptorRef {
    fn get_write(&self, dst_set: Rc<DescriptorSet>, dst_binding: u32) -> WriteDescriptorSet;
    fn get_type(&self) -> vk::DescriptorType;
}

pub struct UniformBufferRef<T: AsStd140>
{
    uniform_buffers: Vec<Rc<UniformBuffer<T>>>,
}

impl<T: AsStd140> UniformBufferRef<T>
{
    pub fn new(uniform_buffers: Vec<Rc<UniformBuffer<T>>>) -> Self {
        Self{
            uniform_buffers,
        }
    }
}

impl<T: AsStd140> DescriptorRef for UniformBufferRef<T>
{
    fn get_write(&self, dst_set: Rc<DescriptorSet>, dst_binding: u32) -> WriteDescriptorSet {
        let mut uniform_buffer_info = vec![];
        for buf in self.uniform_buffers.iter() {
            uniform_buffer_info.push(vk::DescriptorBufferInfo{
                buffer: buf.get_buffer(),
                offset: 0,
                range: buf.len(),
            });
        }
        WriteDescriptorSet::for_buffers(
            vk::DescriptorType::UNIFORM_BUFFER,
            uniform_buffer_info,
            dst_set,
            dst_binding,
        )
    }

    fn get_type(&self) -> vk::DescriptorType {
        vk::DescriptorType::UNIFORM_BUFFER
    }
}

pub struct ComplexUniformBufferRef<T: WriteStd140>
{
    uniform_buffers: Vec<Rc<ComplexUniformBuffer<T>>>,
}

impl<T: WriteStd140> ComplexUniformBufferRef<T>
{
    pub fn new(uniform_buffers: Vec<Rc<ComplexUniformBuffer<T>>>) -> Self {
        Self{
            uniform_buffers,
        }
    }
}

impl<T: WriteStd140> DescriptorRef for ComplexUniformBufferRef<T>
{
    fn get_write(&self, dst_set: Rc<DescriptorSet>, dst_binding: u32) -> WriteDescriptorSet {
        let mut uniform_buffer_info = vec![];
        for buf in self.uniform_buffers.iter() {
            uniform_buffer_info.push(vk::DescriptorBufferInfo{
                buffer: buf.get_buffer(),
                offset: 0,
                range: buf.len(),
            });
        }
        WriteDescriptorSet::for_buffers(
            vk::DescriptorType::UNIFORM_BUFFER,
            uniform_buffer_info,
            dst_set,
            dst_binding,
        )
    }

    fn get_type(&self) -> vk::DescriptorType {
        vk::DescriptorType::UNIFORM_BUFFER
    }
}

pub struct TextureRef {
    textures: Vec<Rc<Texture>>,
}

impl TextureRef {
    pub fn new(textures: Vec<Rc<Texture>>) -> Self {
        Self{
            textures,
        }
    }
}

impl DescriptorRef for TextureRef {
    fn get_write(&self, dst_set: Rc<DescriptorSet>, dst_binding: u32) -> WriteDescriptorSet {
        let mut texture_info = vec![];
        for tex in self.textures.iter() {
            texture_info.push(vk::DescriptorImageInfo{
                sampler: vk::Sampler::null(),
                image_view: tex.image_view.view,
                image_layout: tex.image.layout,
            });
        }
        WriteDescriptorSet::for_images(
            vk::DescriptorType::SAMPLED_IMAGE,
            texture_info,
            dst_set,
            dst_binding,
        )
    }

    fn get_type(&self) -> vk::DescriptorType {
        vk::DescriptorType::SAMPLED_IMAGE
    }
}

pub struct CombinedRef {
    samplers: Vec<Rc<Sampler>>,
    textures: Vec<Rc<Texture>>,
}

impl CombinedRef {
    pub fn new(sampler: Rc<Sampler>, textures: Vec<Rc<Texture>>) -> Self {
        Self{
            samplers: vec![sampler],
            textures,
        }
    }

    pub fn new_per(samplers: Vec<Rc<Sampler>>, textures: Vec<Rc<Texture>>) -> Result<Self> {
        if samplers.len() != textures.len() {
            Err(Error::Generic(format!("{} samplers provided for {} textures", samplers.len(), textures.len())))
        } else {
            Ok(Self{
                samplers,
                textures,
            })
        }
    }
}

impl DescriptorRef for CombinedRef {
    fn get_write(&self, dst_set: Rc<DescriptorSet>, dst_binding: u32) -> WriteDescriptorSet {
        let mut texture_info = vec![];
        for (i, tex) in self.textures.iter().enumerate() {
            texture_info.push(vk::DescriptorImageInfo{
                sampler: if self.samplers.len() == 1 {
                    self.samplers[0].sampler
                } else {
                    self.samplers[i].sampler
                },
                image_view: tex.image_view.view,
                image_layout: tex.image.layout,
            });
        }
        WriteDescriptorSet::for_images(
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            texture_info,
            dst_set,
            dst_binding,
        )
    }

    fn get_type(&self) -> vk::DescriptorType {
        vk::DescriptorType::COMBINED_IMAGE_SAMPLER
    }
}

pub struct InputAttachmentRef {
    texture: Rc<Texture>,
    layout: vk::ImageLayout,
}

impl InputAttachmentRef {
    pub fn new(texture: Rc<Texture>, layout: vk::ImageLayout) -> Self {
        Self{
            texture,
            layout,
        }
    }
}

impl DescriptorRef for InputAttachmentRef {
    fn get_write(&self, dst_set: Rc<DescriptorSet>, dst_binding: u32) -> WriteDescriptorSet {
        let texture_info = vec![vk::DescriptorImageInfo{
            sampler: vk::Sampler::null(),
            image_view: self.texture.image_view.view,
            image_layout: self.layout,
        }];
        WriteDescriptorSet::for_images(
            vk::DescriptorType::INPUT_ATTACHMENT,
            texture_info,
            dst_set,
            dst_binding,
        )
    }

    fn get_type(&self) -> vk::DescriptorType {
        vk::DescriptorType::INPUT_ATTACHMENT
    }
}

#[derive(Clone, Debug)]
pub struct DescriptorBindings {
    bindings: Vec<(vk::DescriptorSetLayoutBinding, vk::DescriptorBindingFlags)>,
    next_binding_id: u32,
}

impl DescriptorBindings {
    pub fn new() -> Self {
        Self{
            bindings: Vec::new(),
            next_binding_id: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    pub fn with_binding(
        mut self,
        descriptor_type: vk::DescriptorType,
        descriptor_count: u32,
        stage_flags: vk::ShaderStageFlags,
        allow_partial: bool,
    ) -> Self {
        let binding_id = self.next_binding_id;
        self.next_binding_id += 1;
        self.with_exact_binding(
            binding_id,
            descriptor_type,
            descriptor_count,
            stage_flags,
            allow_partial,
        )
    }

    pub fn with_exact_binding(
        mut self,
        binding: u32,
        descriptor_type: vk::DescriptorType,
        descriptor_count: u32,
        stage_flags: vk::ShaderStageFlags,
        allow_partial: bool,
    ) -> Self {
        let flags = if allow_partial {
            vk::DescriptorBindingFlags::PARTIALLY_BOUND
                | vk::DescriptorBindingFlags::PARTIALLY_BOUND_EXT
        } else {
            vk::DescriptorBindingFlags::empty()
        };

        self.bindings.push(
            (
                vk::DescriptorSetLayoutBinding{
                    binding,
                    descriptor_type,
                    descriptor_count,
                    stage_flags,
                    p_immutable_samplers: ptr::null(),
                },
                flags,
            )
        );
        self
    }
}

#[derive(Debug)]
pub struct DescriptorSetLayout {
    device: Rc<InnerDevice>,
    pub (crate) layout: vk::DescriptorSetLayout,
    bindings: DescriptorBindings,
}

impl DescriptorSetLayout {
    pub fn new(
        device: &Device,
        bindings: DescriptorBindings,
    ) -> Result<Self> {
        let (binding_set, binding_flags) = {
            let mut binding_sets = vec![];
            let mut binding_flags = vec![];
            for (binding, flags) in bindings.bindings.iter() {
                dbg!(binding);
                dbg!(flags);
                binding_sets.push(binding.clone());
                binding_flags.push(*flags);
            }
            (binding_sets, binding_flags)
        };
        let binding_flags_info = vk::DescriptorSetLayoutBindingFlagsCreateInfo{
            s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
            p_next: ptr::null(),
            binding_count: binding_flags.len() as u32,
            p_binding_flags: binding_flags.as_ptr(),
        };
        let p_next: *const c_void = ((&binding_flags_info) as *const _) as *const c_void;
        let layout_info = vk::DescriptorSetLayoutCreateInfo{
            s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            p_next,
            flags: vk::DescriptorSetLayoutCreateFlags::empty(),
            binding_count: binding_set.len() as u32,
            p_bindings: binding_set.as_ptr(),
        };
        dbg!(&layout_info);
        Ok(Self{
            device: device.inner.clone(),
            layout: unsafe {
                let layout = Error::wrap_result(
                    device.inner.device.create_descriptor_set_layout(&layout_info, None),
                    "Failed to create descriptor set layout",
                )?;
                dbg!(&layout);
                layout
            },
            bindings,
        })
    }
}

impl Drop for DescriptorSetLayout {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_descriptor_set_layout(self.layout, None);
        }
    }
}

#[derive(Debug)]
pub struct DescriptorPool {
    name: String,
    device: Rc<InnerDevice>,
    pools: Vec<vk::DescriptorPool>,
    pool_sizes: HashMap<vk::DescriptorType, u32>,
    max_sets: u32,
    sets: Vec<Rc<DescriptorSet>>,
}

impl DescriptorPool {
    pub fn new(
        name: &str,
        device: &Device,
        pool_sizes: HashMap<vk::DescriptorType, u32>,
        max_sets: u32,
    ) -> Result<Self> {
        let mut this = Self{
            name: name.to_owned(),
            device: device.inner.clone(),
            pools: Vec::new(),
            pool_sizes,
            max_sets,
            sets: Vec::new(),
        };
        this.grow()?;
        Ok(this)
    }

    pub fn reset(&mut self) -> Result<()> {
        //println!("Resetting descriptor pool {}", self.name);
        let mut i = 0;
        for set in &self.sets {
            if Rc::strong_count(set) > 1 {
                return Err(
                    Error::Generic(format!("Descriptor set {} has external references, but we are resetting its pool!", i))
                );
            }
            i += 1;
        }
        self.sets.clear();
        unsafe {
            if self.pools.len() > 1 {
                for pool in self.pools[1..].iter() {
                    self.device.device.destroy_descriptor_pool(*pool, None);
                }
            }
            self.pools.truncate(1);
            Error::wrap_result(
                self.device.device.reset_descriptor_pool(
                    self.pools[0],
                    vk::DescriptorPoolResetFlags::empty(),
                ),
                "Failed to reset descriptor pool",
            )?;
        }
        Ok(())
    }

    fn grow(&mut self) -> Result<()> {
        let mut sizes = vec![];
        for (ty, descriptor_count) in self.pool_sizes.iter() {
            sizes.push(vk::DescriptorPoolSize{
                ty: *ty,
                descriptor_count: *descriptor_count,
            });
        }
        let create_info = vk::DescriptorPoolCreateInfo{
            s_type: vk::StructureType::DESCRIPTOR_POOL_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::DescriptorPoolCreateFlags::empty(),
            max_sets: self.max_sets,
            pool_size_count: sizes.len() as u32,
            p_pool_sizes: sizes.as_ptr(),
        };
        self.pools.push(unsafe {
            Error::wrap_result(
                self.device.device.create_descriptor_pool(&create_info, None),
                "Failed to create descriptor pool when growing",
            )?
        });
        Ok(())
    }

    pub fn create_descriptor_sets(
        &mut self,
        count: usize,
        layout: &DescriptorSetLayout,
        items: &[Rc<dyn DescriptorRef>],
    ) -> Result<Vec<Rc<DescriptorSet>>> {
        if items.len() != layout.bindings.len() {
            return Err(Error::Generic(format!(
                "Provided {} items for a layout that takes {} items",
                items.len(),
                layout.bindings.len(),
            )));
        }

        let sets = self.allocate(count, layout)?;
        for set in sets.iter() {
            let mut writes = vec![];
            // These have to stick around until the call to update_descriptor_sets() returns.
            let mut writers = vec![];
            for (binding, item) in items.iter().enumerate() {
                writers.push(item.get_write(Rc::<DescriptorSet>::clone(set), binding as u32));
                writes.push(writers.last().unwrap().get());
                set.add_dependency(Rc::<dyn DescriptorRef>::clone(item));
            }
            if DEBUG_DESCRIPTOR_SETS {
                println!("Performing {} writes to descriptor set {:?}...", writes.len(), set.inner);
                for write in writes.iter() {
                    println!(
                        "\tset: {:?}\n\tbinding: {}\n\ttype: {:?}\n\tcount: {}",
                        write.dst_set,
                        write.dst_binding,
                        write.descriptor_type,
                        write.descriptor_count,
                    );
                }
            }
            unsafe {
                self.device.device.update_descriptor_sets(&writes, &[]);
            }
            self.sets.push(Rc::clone(set));
        }
        Ok(sets)
    }

    fn allocate(
        &mut self,
        count: usize,
        layout: &DescriptorSetLayout,
    ) -> Result<Vec<Rc<DescriptorSet>>> {
        self.allocate_internal(count, layout, true)
    }

    fn allocate_internal(
        &mut self,
        count: usize,
        layout: &DescriptorSetLayout,
        retry: bool,
    ) -> Result<Vec<Rc<DescriptorSet>>> {
        let mut layouts: Vec<vk::DescriptorSetLayout> = vec![];
        for _ in 0..count {
            layouts.push(layout.layout);
        }

        let info = vk::DescriptorSetAllocateInfo{
            s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
            p_next: ptr::null(),
            descriptor_pool: *self.pools.last().unwrap(),
            descriptor_set_count: layouts.len() as u32,
            p_set_layouts: layouts.as_ptr(),
        };

        match unsafe { self.device.device.allocate_descriptor_sets(&info) } {
            Ok(vk_sets) => Ok({
                let mut sets = Vec::new();
                for set in vk_sets {
                    sets.push(Rc::new(DescriptorSet{
                        device: Rc::clone(&self.device),
                        inner: set,
                        dependencies: RefCell::new(Vec::new()),
                    }));
                }
                sets
            }),
            Err(vk::Result::ERROR_FRAGMENTED_POOL) | Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY) => {
                if retry {
                    self.grow()?;
                    self.allocate_internal(count, layout, false)
                } else {
                    Err(Error::Generic("Failed second attempt at allocating descriptor set".to_owned()))
                }
            },
            Err(e) => Err(Error::Generic(format!("Cannot allocate descriptor set: {:?}", e))),
        }
    }
}

impl Drop for DescriptorPool {
    fn drop(&mut self) {
        let mut i = 0;
        for set in &self.sets {
            if Rc::strong_count(set) > 1 {
                println!("Descriptor set {} has external references, but we are destroying its pool!", i);
            }
            i += 1;
        }
        unsafe {
            for pool in self.pools.iter() {
                self.device.device.destroy_descriptor_pool(*pool, None);
            }
        }
    }
}

pub struct DescriptorSet {
    device: Rc<InnerDevice>,
    pub (crate) inner: vk::DescriptorSet,
    dependencies: RefCell<Vec<Rc<dyn DescriptorRef>>>,
}

impl DescriptorSet {
    fn add_dependency(&self, item: Rc<dyn DescriptorRef>) {
        self.dependencies.borrow_mut().push(item);
    }
}

impl std::fmt::Debug for DescriptorSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DescriptorSet")
            .field("device", &self.device)
            .field("inner", &self.inner)
            .field("dependencies", &format!("[{} items]", self.dependencies.borrow().len()))
            .finish()
    }
}
