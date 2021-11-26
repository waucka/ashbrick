//! This module contains types for working with textures and samplers.
//! It's relatively high-level, so users who want to load some textures
//! will want to use this more than the image module.

use ash::vk;
use image::GenericImageView;
use log::trace;

#[cfg(feature = "egui")]
use std::sync::Arc;
use std::rc::Rc;
use std::ptr;
use std::path::Path;
use std::cmp::max;

use super::{Device, InnerDevice, Queue};
use super::command_buffer::CommandPool;
use super::image::{Image, ImageView, ImageBuilder};
use super::buffer::UploadSourceBuffer;

use super::errors::{Error, Result};

pub struct Texture {
    pub (crate) image: Rc<Image>,
    pub (crate) image_view: Rc<ImageView>,
    mip_levels: u32,
    name: String,
}

impl Texture {
    pub fn from_float_tex(
        device: &Device,
        name: &str,
        path: &Path,
        pool: Rc<CommandPool>,
        queue: &Queue,
    ) -> Result<Self> {
        let data_bytes = Error::wrap_io(
            std::fs::read(path),
            &format!("Failed to read image data from {}", path.display()),
        )?;
        if data_bytes.len() % 4 != 0 {
            return Err(Error::InvalidTextureSize{
                data_len: data_bytes.len(),
                required_multiple: 4,
            });
        }
        let num_values = data_bytes.len() / 4;
        let mip_levels = 1;
        /*let data_float = Vec::with_capacity(num_values);
        for i in 0..num_values {
            data_float.push(f32::from_le_bytes([
                data_bytes[i + 0],
                data_bytes[i + 1],
                data_bytes[i + 2],
                data_bytes[i + 3],
            ]));
        }*/
        let image_size =
            (std::mem::size_of::<u8>() * data_bytes.len()) as vk::DeviceSize;
        let upload_buffer = UploadSourceBuffer::new(device, "temp-upload-source-buffer", image_size)?;
        upload_buffer.copy_data(&data_bytes)?;

        let mut image = Image::new(
            device,
            ImageBuilder::new1d(name, num_values as usize)
                .with_mip_levels(mip_levels)
                .with_num_samples(vk::SampleCountFlags::TYPE_1)
                .with_format(vk::Format::R32G32B32_SFLOAT)
                .with_tiling(vk::ImageTiling::OPTIMAL)
                .with_usage(
                    vk::ImageUsageFlags::TRANSFER_SRC |
                    vk::ImageUsageFlags::TRANSFER_DST |
                    vk::ImageUsageFlags::SAMPLED,
                )
                .with_required_memory_properties(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                .with_sharing_mode(vk::SharingMode::EXCLUSIVE)
        )?;

        image.transition_layout(
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            mip_levels,
            Rc::clone(&pool),
            queue,
        )?;

        unsafe {
            Image::copy_buffer_no_deps(&upload_buffer, &image, Rc::clone(&pool), queue)?;
        }

        image.transition_layout(
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            mip_levels,
            pool,
            queue,
        )?;

        let image_view = Rc::new(ImageView::from_image(
            &image,
            vk::ImageAspectFlags::COLOR,
            mip_levels,
        )?);

        let image = Rc::new(image);

        Ok(Self{
            image,
            image_view,
            mip_levels,
            name: String::from(name),
        })
    }

    pub fn from_exr_1d(
        device: &Device,
        name: &str,
        path: &Path,
        pool: Rc<CommandPool>,
        queue: &Queue,
    ) -> Result<Self> {
        let image_res = {
            use exr::prelude::*;
            read()
                .no_deep_data()
                .largest_resolution_level()
                .specific_channels()
                .required("R")
                .required("G")
                .required("B")
                .collect_pixels(
                    |resolution, _| {
                        let num_values = resolution.width() * resolution.height() * 3;
                        let empty_image = vec![0.0; num_values];
                        empty_image
                    },
                    |pixel_vector, position, (r, g, b): (f32, f32, f32)| {
                        /*trace!(
                            "[{}, {}] = ({}, {}, {})",
                            position.x(), position.y(),
                            r, g, b,
                        );*/
                        let y = position.y() + 1;
                        pixel_vector[y * position.x() * 3    ] = r;
                        pixel_vector[y * position.x() * 3 + 1] = g;
                        pixel_vector[y * position.x() * 3 + 2] = b;
                    },
                )
                .all_layers()
                .all_attributes()
                .from_file(path)
        };
        let image = match image_res {
            Ok(v) => v,
            Err(e) => return Err(Error::external(e, "Failed to load OpenEXR file")),
        };

        let first_layer = match image.layer_data.first() {
            Some(layer) => layer,
            None => return Err(Error::Generic(format!("OpenEXR image {} contains no layers", path.display()))),
        };
        let pixels = &first_layer.channel_data.pixels;

        let image_size =
            (std::mem::size_of::<f32>() * pixels.len()) as vk::DeviceSize;
        let upload_buffer = UploadSourceBuffer::new(device, "temp-upload-source-buffer", image_size)?;
        upload_buffer.copy_data(pixels)?;

        let mip_levels = 1;
        let mut image = Image::new(
            device,
            ImageBuilder::new1d(name, pixels.len() / 3)
                .with_mip_levels(mip_levels)
                .with_num_samples(vk::SampleCountFlags::TYPE_1)
                .with_format(vk::Format::R32G32B32_SFLOAT)
                .with_tiling(vk::ImageTiling::OPTIMAL)
                .with_usage(
                    vk::ImageUsageFlags::TRANSFER_SRC |
                    vk::ImageUsageFlags::TRANSFER_DST |
                    vk::ImageUsageFlags::SAMPLED,
                )
                .with_required_memory_properties(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                .with_sharing_mode(vk::SharingMode::EXCLUSIVE)
        )?;

        image.transition_layout(
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            mip_levels,
            Rc::clone(&pool),
            queue,
        )?;

        unsafe {
            Image::copy_buffer_no_deps(&upload_buffer, &image, Rc::clone(&pool), queue)?;
        }

        image.transition_layout(
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            mip_levels,
            pool,
            queue,
        )?;

        let image_view = Rc::new(ImageView::from_image(
            &image,
            vk::ImageAspectFlags::COLOR,
            mip_levels,
        )?);

        let image = Rc::new(image);

        Ok(Self{
            image,
            image_view,
            mip_levels,
            name: String::from(name),
        })
    }

    #[cfg(feature = "egui")]
    pub fn from_egui(
        device: &Device,
        egui_texture: &Arc<egui::paint::Texture>,
        name: &str,
        pool: Rc<CommandPool>,
        queue: &Queue,
    ) -> anyhow::Result<Self> {
        let (image_width, image_height) = (egui_texture.width as u32, egui_texture.height as u32);
        let image_size =
            (std::mem::size_of::<u8>() as u32 * image_width * image_height * 4) as vk::DeviceSize;
        let mip_levels = 1;

        let upload_buffer = UploadSourceBuffer::new(device, "temp-upload-source-buffer", image_size)?;
        let srgba_pixels: Vec<egui::paint::Color32> = egui_texture.srgba_pixels().collect();
        upload_buffer.copy_data(&srgba_pixels)?;

        let mut image = Image::new(
            device,
            ImageBuilder::new2d(image_width as usize, image_height as usize)
                .with_mip_levels(mip_levels)
                .with_num_samples(vk::SampleCountFlags::TYPE_1)
                .with_format(vk::Format::R8G8B8A8_SRGB)
                .with_tiling(vk::ImageTiling::OPTIMAL)
                .with_usage(
                    vk::ImageUsageFlags::TRANSFER_SRC |
                    vk::ImageUsageFlags::TRANSFER_DST |
                    vk::ImageUsageFlags::SAMPLED,
                )
                .with_required_memory_properties(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                .with_sharing_mode(vk::SharingMode::EXCLUSIVE)
        )?;

        image.transition_layout(
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            mip_levels,
            pool,
            queue,
        )?;

        unsafe {
            Image::copy_buffer_no_deps(&upload_buffer, &image)?;
        }

        image.transition_layout(
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            mip_levels,
            pool,
            queue,
        )?;

        let image_view = Rc::new(ImageView::from_image(
            &image,
            vk::ImageAspectFlags::COLOR,
            mip_levels,
        )?);

        let image = Rc::new(image);

        Ok(Self{
            image,
            image_view,
            mip_levels,
            name: String::from(name),
        })
    }

    pub fn from_image(
        image: Rc<Image>,
        image_view: Rc<ImageView>,
        mip_levels: u32,
        name: &str,
    ) -> Result<Self> {
        Ok(Self{
            image,
            image_view,
            mip_levels,
            name: String::from(name),
        })
    }

    pub fn get_image(&self) -> Rc<Image> {
        Rc::clone(&self.image)
    }

    pub fn get_image_view(&self) -> Rc<ImageView> {
        Rc::clone(&self.image_view)
    }

    pub fn get_image_debug_str(&self) -> String {
        format!("{:?}", self.image.img)
    }

    pub fn from_bytes(
        device: &Device,
        name: &str,
        bytes: &[u8],
        srgb: bool,
        mipmapped: bool,
        pool: Rc<CommandPool>,
        queue: &Queue,
    ) -> Result<Self> {
        let start = std::time::Instant::now();
        let image_object = match image::load_from_memory(bytes) {
            Ok(v) => v,
            Err(e) => return Err(Error::external(e, &format!("Failed to load image from {} bytes", bytes.len()))),
        };
        trace!("Loaded {} bytes in {}ms", bytes.len(), start.elapsed().as_millis());
        Self::from_dynamic_image(device, name, &image_object, srgb, mipmapped, pool, queue)
    }

    pub fn from_file(
        device: &Device,
        name: &str,
        image_path: &Path,
        srgb: bool,
        mipmapped: bool,
        pool: Rc<CommandPool>,
        queue: &Queue,
    ) -> Result<Self> {
        let start = std::time::Instant::now();
        let image_object = match image::open(image_path) {
            Ok(v) => v,
            Err(e) => return Err(Error::external(e, &format!("Failed to load image {}", image_path.display()))),
        };
        trace!("Loaded {} in {}ms", image_path.display(), start.elapsed().as_millis());
        Self::from_dynamic_image(device, name, &image_object, srgb, mipmapped, pool, queue)
    }

    pub fn from_dynamic_image(
        device: &Device,
        name: &str,
        image_object: &image::DynamicImage,
        srgb: bool,
        mipmapped: bool,
        pool: Rc<CommandPool>,
        queue: &Queue,
    ) -> Result<Self> {
        use image::DynamicImage::*;
        use image::buffer::ConvertBuffer;
        match image_object {
            ImageRgb8(img) => {
                let image_object = ImageRgba8(img.convert());
                Self::from_dynamic_image_internal(device, name, &image_object, srgb, mipmapped, pool, queue)
            },
            ImageRgb16(img) => {
                let image_object = ImageRgba16(img.convert());
                Self::from_dynamic_image_internal(device, name, &image_object, srgb, mipmapped, pool, queue)
            },
            ImageBgr8(img) => {
                let image_object = ImageRgba8(img.convert());
                Self::from_dynamic_image_internal(device, name, &image_object, srgb, mipmapped, pool, queue)
            },
            image_object => Self::from_dynamic_image_internal(device, name, image_object, srgb, mipmapped, pool, queue),
        }
    }

    fn from_dynamic_image_internal(
        device: &Device,
        name: &str,
        image_object: &image::DynamicImage,
        srgb: bool,
        mipmapped: bool,
        pool: Rc<CommandPool>,
        queue: &Queue,
    ) -> Result<Self> {
        use image::DynamicImage::*;
        let (bytes_per_channel, channels_per_pixel, format) = match &image_object {
            ImageLuma8(_) => (1, 1, if srgb { vk::Format::R8_SRGB } else { vk::Format::R8_UNORM }),
            ImageLumaA8(_) => (1, 2, if srgb { vk::Format::R8G8_SRGB } else { vk::Format::R8G8_UNORM }),
            ImageRgba8(_) => (1, 4, if srgb { vk::Format::R8G8B8A8_SRGB } else { vk::Format::R8G8B8A8_UNORM }),
            ImageBgra8(_) => (1, 4, if srgb { vk::Format::R8G8B8A8_SRGB } else { vk::Format::R8G8B8A8_UNORM }),
            ImageLuma16(_) => (2, 1, if srgb { return Err(Error::SrgbNotAvailable) } else { vk::Format::R16_UNORM }),
            ImageLumaA16(_) => (2, 2, if srgb { return Err(Error::SrgbNotAvailable) } else { vk::Format::R16G16_UNORM }),
            ImageRgba16(_) => (2, 4, if srgb { return Err(Error::SrgbNotAvailable) } else { vk::Format::R16G16B16A16_UNORM }),
            ImageRgb8(_) | ImageRgb16(_) | ImageBgr8(_) => return Err(Error::internal("Bad image passed to from_image_internal")),
        };
        let (width, height) = (image_object.width(), image_object.height());

        let image_data = ImageData{
            data: image_object.as_bytes(),
            bytes_per_channel,
            channels_per_pixel,
            width,
            height,
        };

        Self::from_image_data(
            device,
            name,
            &image_data,
            format,
            mipmapped,
            pool,
            queue,
        )
    }

    pub fn from_image_data(
        device: &Device,
        name: &str,
        image_data: &ImageData,
        format: vk::Format,
        mipmapped: bool,
        pool: Rc<CommandPool>,
        queue: &Queue,
    ) -> Result<Self> {
        let (image_width, image_height) = (image_data.width, image_data.height);
        let size_of_pixel = image_data.bytes_per_channel * image_data.channels_per_pixel;
        let image_size =
            (std::mem::size_of::<u8>() as u32 * image_width * image_height * size_of_pixel) as vk::DeviceSize;
        let mip_levels = if mipmapped {
            ((max(image_width, image_height) as f32)
             .log2()
             .floor() as u32) + 1
        } else {
            1
        };

        if image_size == 0 {
            panic!("Invalid image size");
        }

        let upload_buffer = UploadSourceBuffer::new(device, "temp-upload-source-buffer", image_size)?;
        upload_buffer.copy_data(image_data.data)?;

        let mut image = Image::new(
            device,
            if image_width == 1 || image_height == 1 {
                ImageBuilder::new1d(name, image_width as usize)
            } else {
                ImageBuilder::new2d(name, image_width as usize, image_height as usize)
            }
            .with_mip_levels(mip_levels)
                .with_num_samples(vk::SampleCountFlags::TYPE_1)
                .with_format(format)
                .with_tiling(vk::ImageTiling::OPTIMAL)
                .with_usage(
                    vk::ImageUsageFlags::TRANSFER_SRC |
                    vk::ImageUsageFlags::TRANSFER_DST |
                    vk::ImageUsageFlags::SAMPLED,
                )
                .with_required_memory_properties(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                .with_sharing_mode(vk::SharingMode::EXCLUSIVE)
        )?;

        image.transition_layout(
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            mip_levels,
            Rc::clone(&pool),
            queue,
        )?;

        unsafe {
            Image::copy_buffer_no_deps(
                &upload_buffer,
                &image,
                Rc::clone(&pool),
                queue,
            )?;
        }

        if mipmapped {
            let start = std::time::Instant::now();
            image.generate_mipmaps(
                mip_levels,
                pool,
                queue,
            )?;
            trace!("Generated mipmaps in {}ms", start.elapsed().as_millis());
        } else {
            image.transition_layout(
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                mip_levels,
                pool,
                queue,
            )?;
        }

        let image_view = Rc::new(ImageView::from_image(
            &image,
            vk::ImageAspectFlags::COLOR,
            mip_levels,
        )?);

        let image = Rc::new(image);

        Ok(Self{
            image,
            image_view,
            mip_levels,
            name: String::from(name),
        })
    }

    pub fn get_mip_levels(&self) -> u32 {
        self.mip_levels
    }

    pub fn get_descriptor_info(&self, sampler: &Sampler) -> vk::DescriptorImageInfo {
        vk::DescriptorImageInfo{
            sampler: sampler.sampler,
            image_view: self.image_view.view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        }
    }

    pub fn get_extent(&self) -> vk::Extent3D {
        self.image.extent
    }
}

impl super::NamedResource for Texture {
    fn name(&self) -> &str {
        &self.name
    }
}

pub struct ImageData<'a> {
    pub data: &'a [u8],
    pub bytes_per_channel: u32,
    pub channels_per_pixel: u32,
    pub width: u32,
    pub height: u32,
}

pub struct Sampler {
    device: Rc<InnerDevice>,
    pub (crate) sampler: vk::Sampler,
}

impl Sampler {
    pub fn new(
        device: &Device,
        mip_levels: u32,
        min_filter: vk::Filter,
        mag_filter: vk::Filter,
        mipmap_mode: vk::SamplerMipmapMode,
        address_mode_u: vk::SamplerAddressMode,
        address_mode_v: vk::SamplerAddressMode,
        address_mode_w: vk::SamplerAddressMode,
        max_anisotropy: u32,
    ) -> Result<Self> {
        let sampler_create_info = vk::SamplerCreateInfo{
            s_type: vk::StructureType::SAMPLER_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::SamplerCreateFlags::empty(),
            min_filter,
            mag_filter,
            mipmap_mode,
            address_mode_u,
            address_mode_v,
            address_mode_w,
            mip_lod_bias: 0.0,
            anisotropy_enable: if max_anisotropy > 0 {
                vk::TRUE
            } else {
                vk::FALSE
            },
            max_anisotropy: max_anisotropy as f32,
            compare_enable: vk::FALSE,
            compare_op: vk::CompareOp::ALWAYS,
            min_lod: 0.0,
            max_lod: mip_levels as f32,
            border_color: vk::BorderColor::INT_OPAQUE_BLACK,
            unnormalized_coordinates: vk::FALSE,
        };

        unsafe {
            Ok(Self {
                device: device.inner.clone(),
                sampler: Error::wrap_result(
                    device.inner.device.create_sampler(&sampler_create_info, None),
                    "Failed to create sampler",
                )?,
            })
        }
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_sampler(self.sampler, None);
        }
    }
}

pub struct CombinedTexture {
    sampler: Sampler,
    texture: Texture,
}

impl CombinedTexture {
    pub fn new(
        sampler: Sampler,
        texture: Texture,
    ) -> Self {
        Self{
            sampler,
            texture,
        }
    }

    pub fn get_descriptor_info(&self) -> vk::DescriptorImageInfo {
        vk::DescriptorImageInfo{
            sampler: self.sampler.sampler,
            image_view: self.texture.image_view.view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        }
    }
}
