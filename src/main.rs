// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

// This example demonstrates how to use the compute capabilities of Vulkan.
//
// While graphics cards have traditionally been used for graphical operations, over time they have
// been more or more used for general-purpose operations as well. This is called "General-Purpose
// GPU", or *GPGPU*. This is what this example demonstrates.

use std::sync::Arc;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        DispatchIndirectCommand,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo,
        QueueFlags,
    },
    instance::{Instance, InstanceCreateInfo},
    memory::{
        allocator::{
            AllocationCreateInfo, MemoryAllocatePreference, MemoryTypeFilter, MemoryUsage,
            StandardMemoryAllocator,
        },
        MemoryPropertyFlags,
    },
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout},
    sync::{self, GpuFuture},
    VulkanLibrary,
};

fn main() {
    // As with other examples, the first step is to create an instance.
    let library = VulkanLibrary::new().unwrap();
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            enumerate_portability: true,
            ..Default::default()
        },
    )
    .unwrap();

    // Choose which physical device to use.
    let device_extensions = DeviceExtensions {
        khr_storage_buffer_storage_class: true,
        ..DeviceExtensions::empty()
    };
    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| p.supported_extensions().contains(&device_extensions))
        .filter_map(|p| {
            // The Vulkan specs guarantee that a compliant implementation must provide at least one
            // queue that supports compute operations.
            p.queue_family_properties()
                .iter()
                .position(|q| q.queue_flags.intersects(QueueFlags::COMPUTE))
                .map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
            _ => 5,
        })
        .unwrap();

    println!(
        "Using device: {} (type: {:?})",
        physical_device.properties().device_name,
        physical_device.properties().device_type,
    );

    // Now initializing the device.
    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            enabled_extensions: device_extensions,
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    )
    .unwrap();

    // Since we can request multiple queues, the `queues` variable is in fact an iterator. In this
    // example we use only one queue, so we just retrieve the first and only element of the
    // iterator and throw it away.
    let queue = queues.next().unwrap();

    // Now let's get to the actual example.
    //
    // What we are going to do is very basic: we are going to fill a buffer with 64k integers and
    // ask the GPU to multiply each of them by 12.
    //
    // GPUs are very good at parallel computations (SIMD-like operations), and thus will do this
    // much more quickly than a CPU would do. While a CPU would typically multiply them one by one
    // or four by four, a GPU will do it by groups of 32 or 64.
    //
    // Note however that in a real-life situation for such a simple operation the cost of accessing
    // memory usually outweighs the benefits of a faster calculation. Since both the CPU and the
    // GPU will need to access data, there is no other choice but to transfer the data through the
    // slow PCI express bus.

    // We need to create the compute pipeline that describes our operation.
    //
    // If you are familiar with graphics pipeline, the principle is the same except that compute
    // pipelines are much simpler to create.
    mod cs1 {
        vulkano_shaders::shader! {
            ty: "compute",
            spirv_version: "1.5",
            path: "src/multi_radixsort_histograms.comp"
        }
    }
    mod cs2 {
        vulkano_shaders::shader! {
            ty: "compute",
            spirv_version: "1.5",
            path: "src/multi_radixsort.comp"
        }
    }
    let pipeline = {
        let hist = vulkano::pipeline::ComputePipeline::new(
            device.clone(),
            cs1::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap(),
            &(),
            None,
            |_| {},
        )
        .expect("Failed to create compute shader");

        let radix = vulkano::pipeline::ComputePipeline::new(
            device.clone(),
            cs2::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap(),
            &(),
            None,
            |_| {},
        )
        .expect("Failed to create compute shader");
        (hist, radix)
    };

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone());
    let command_buffer_allocator =
        StandardCommandBufferAllocator::new(device.clone(), Default::default());

    const NUM_ELEMENTS: u32 = 1_000_000;
    const NUM_BLOCKS_PER_WORKGROUP: u32 = 32;
    const WORK_GROUP_SIZE: u32 = 256;
    const DISPATCH_SIZE: u32 = NUM_ELEMENTS.div_ceil(NUM_BLOCKS_PER_WORKGROUP);
    const NUM_WORKGROUPS: u32 = DISPATCH_SIZE.div_ceil(WORK_GROUP_SIZE);
    println!(
        "NUM_ELEMENTS: {}, NUM_BLOCKS_PER_WORKGROUP: {}, WORK_GROUP_SIZE: {}, DISPATCH_SIZE: {}, NUM_WORKGROUPS: {}",
        NUM_ELEMENTS, NUM_BLOCKS_PER_WORKGROUP, WORK_GROUP_SIZE, DISPATCH_SIZE, NUM_WORKGROUPS
    );
    let mut numbers: Vec<_> = (0..NUM_ELEMENTS).map(|_| rand::random::<u32>()).collect();
    let mut payloads: Vec<_> = (0..NUM_ELEMENTS).map(|_| rand::random::<u32>()).collect();
    // We start by creating the buffer that will store the data.
    let mut buffer0: Subbuffer<[u32]> = Buffer::from_iter(
        &memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            usage: MemoryUsage::Upload,
            ..Default::default()
        },
        // Iterator that produces the data.
        numbers.iter().cloned(),
    )
    .unwrap();

    let mut buffer1: Subbuffer<[u32]> = Buffer::new_unsized(
        &memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            usage: MemoryUsage::DeviceOnly,
            ..Default::default()
        },
        NUM_ELEMENTS as u64,
    )
    .unwrap();
let mut payloads0: Subbuffer<[u32]> = Buffer::from_iter(
    &memory_allocator.clone(),
    BufferCreateInfo {
        usage: BufferUsage::STORAGE_BUFFER,
        ..Default::default()
    },
    AllocationCreateInfo {
        usage: MemoryUsage::Upload,
        ..Default::default()
    },
    // Iterator that produces the data.
    payloads.iter().cloned(),
)
.unwrap();

let mut payloads1: Subbuffer<[u32]> = Buffer::new_unsized(
    &memory_allocator.clone(),
    BufferCreateInfo {
        usage: BufferUsage::STORAGE_BUFFER,
        ..Default::default()
    },
    AllocationCreateInfo {
        usage: MemoryUsage::DeviceOnly,
        ..Default::default()
    },
    NUM_ELEMENTS as u64,
)
.unwrap();

    let histograms: Subbuffer<[u32]> = Buffer::new_unsized(
        &memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            usage: MemoryUsage::DeviceOnly,
            ..Default::default()
        },
        NUM_WORKGROUPS as u64 * 256,
    )
    .unwrap();

    let pc: Subbuffer<cs1::PC> = Buffer::new_sized(
        &memory_allocator,
        BufferCreateInfo {
            usage: BufferUsage::UNIFORM_BUFFER | BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            usage: MemoryUsage::Upload,
            ..Default::default()
        },
    )
    .unwrap();
    {
        let mut pc_content = pc.write().unwrap();
        pc_content.g_num_elements = NUM_ELEMENTS;
        pc_content.g_num_workgroups = NUM_WORKGROUPS;
    }

    let indirect: Subbuffer<[DispatchIndirectCommand]> = Buffer::from_iter(
        &memory_allocator,
        BufferCreateInfo {
            usage: BufferUsage::INDIRECT_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            usage: MemoryUsage::Upload,
            ..Default::default()
        },
        [DispatchIndirectCommand {
            x: NUM_WORKGROUPS,
            y: 1,
            z: 1,
        }]
        .iter()
        .cloned(),
    )
    .unwrap();

    // In order to let the shader access the buffer, we need to build a *descriptor set* that
    // contains the buffer.
    //
    // The resources that we bind to the descriptor set must match the resources expected by the
    // pipeline which we pass as the first parameter.
    //
    // If you want to run the pipeline on multiple different buffers, you need to create multiple
    // descriptor sets that each contain the buffer you want to run the shader on.

    // In order to execute our operation, we have to build a command buffer.
    let hist_layout = pipeline.0.layout().set_layouts().get(0).unwrap();
    let radix_layout = pipeline.1.layout().set_layouts().get(1).unwrap();
    let mut builder = AutoCommandBufferBuilder::primary(
        &command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    for shift in (0..32).step_by(8) {
        let hist_push = cs1::PushConstants {
            // g_num_elements: NUM_ELEMENTS,
            g_shift: shift,
            // g_num_workgroups: NUM_WORKGROUPS,
            g_num_blocks_per_workgroup: NUM_BLOCKS_PER_WORKGROUP,
        };
        let radix_push = cs2::PushConstants {
            // g_num_elements: NUM_ELEMENTS,
            g_shift: shift,
            // g_num_workgroups: NUM_WORKGROUPS,
            g_num_blocks_per_workgroup: NUM_BLOCKS_PER_WORKGROUP,
        };
        let hist_set = PersistentDescriptorSet::new(
            &descriptor_set_allocator,
            hist_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, buffer0.clone()),
                WriteDescriptorSet::buffer(1, histograms.clone()),
                WriteDescriptorSet::buffer(5, pc.clone()),
            ],
        )
        .unwrap();

        let radix_set = PersistentDescriptorSet::new(
            &descriptor_set_allocator,
            radix_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, buffer0.clone()),
                WriteDescriptorSet::buffer(1, buffer1.clone()),
                WriteDescriptorSet::buffer(3, payloads0.clone()),
                WriteDescriptorSet::buffer(4, payloads1.clone()),
                WriteDescriptorSet::buffer(2, histograms.clone()),
                WriteDescriptorSet::buffer(5, pc.clone()),
            ],
        )
        .unwrap();
        builder
            // The command buffer only does one thing: execute the compute pipeline. This is called a
            // *dispatch* operation.
            //
            // Note that we clone the pipeline and the set. Since they are both wrapped in an `Arc`,
            // this only clones the `Arc` and not the whole pipeline or set (which aren't cloneable
            // anyway). In this example we would avoid cloning them since this is the last time we use
            // them, but in real code you would probably need to clone them.
            .bind_pipeline_compute(pipeline.0.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipeline.0.layout().clone(),
                0,
                hist_set,
            )
            .push_constants(pipeline.0.layout().clone(), 0, hist_push)
            .dispatch_indirect(indirect.clone())
            // .dispatch([NUM_WORKGROUPS, 1, 1])
            .unwrap()
            .bind_pipeline_compute(pipeline.1.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipeline.1.layout().clone(),
                1,
                radix_set,
            )
            .push_constants(pipeline.1.layout().clone(), 0, radix_push)
            .dispatch_indirect(indirect.clone())
            // .dispatch([NUM_WORKGROUPS, 1, 1])
            .unwrap();
        if shift < 24 {
            std::mem::swap(&mut buffer0, &mut buffer1);
            std::mem::swap(&mut payloads0, &mut payloads1);
        }
        // std::mem::swap(&mut buffer0, &mut buffer1);
    }

    // Finish building the command buffer by calling `build`.
    let command_buffer = builder.build().unwrap();

    // Let's execute this command buffer now.
    let future = sync::now(device)
        .then_execute(queue, command_buffer)
        .unwrap()
        // This line instructs the GPU to signal a *fence* once the command buffer has finished
        // execution. A fence is a Vulkan object that allows the CPU to know when the GPU has
        // reached a certain point. We need to signal a fence here because below we want to block
        // the CPU until the GPU has reached that point in the execution.
        .then_signal_fence_and_flush()
        .unwrap();

    // Blocks execution until the GPU has finished the operation. This method only exists on the
    // future that corresponds to a signalled fence. In other words, this method wouldn't be
    // available if we didn't call `.then_signal_fence_and_flush()` earlier. The `None` parameter
    // is an optional timeout.
    //
    // Note however that dropping the `future` variable (with `drop(future)` for example) would
    // block execution as well, and this would be the case even if we didn't call
    // `.then_signal_fence_and_flush()`. Therefore the actual point of calling
    // `.then_signal_fence_and_flush()` and `.wait()` is to make things more explicit. In the
    // future, if the Rust language gets linear types vulkano may get modified so that only
    // fence-signalled futures can get destroyed like this.
    future.wait(None).unwrap();

    // Now that the GPU is done, the content of the buffer should have been modified. Let's check
    // it out. The call to `read()` would return an error if the buffer was still in use by the
    // GPU.
    let mut numbers_payload = numbers.iter().zip(payloads.iter()).map(|(n, p)| (*n, *p)).collect::<Vec<_>>();
    numbers_payload.sort_by_key(|(n, _)| *n);
    numbers.sort();
    let data_buffer_content = buffer1.read().unwrap();
    let payloads_content = payloads1.read().unwrap();
    // println!("numbers: {:?}", &numbers);
    // println!("data_buffer_content: {:?}", &data_buffer_content[..]);
    println!("first 10 numbers: {:?}", &numbers[0..10]);
    println!(
        "first 10 numbers of buffer: {:?}",
        &data_buffer_content[0..10]
    );
    println!(
        "last 10 numbers of buffer: {:?}",
        &data_buffer_content[NUM_ELEMENTS as usize - 10..]
    );
    // data_buffer_content.sort();
    for i in 0..NUM_ELEMENTS as usize {
        assert_eq!(numbers_payload[i].0, data_buffer_content[i]);
        assert_eq!(numbers_payload[i].1, payloads_content[i]);
    }

    println!("Success");
}
