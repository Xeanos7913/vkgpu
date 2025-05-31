#pragma once

#include "VkBootstrap.h"
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_core.h>
#include <typeinfo>
#include <fstream>
#include <iostream>
#include <vector>


struct Init {
    vkb::Instance instance;
    vkb::InstanceDispatchTable inst_disp;
    vkb::Device device;
    vkb::DispatchTable disp;

    struct func {
		void (*func)(void*);
		void* ptr;
    };

	std::vector<func> funcs;

    template<typename T>
	static void invokeFunc(void* p) {
		static_cast<T*>(p)->destructor();
	}

	template<typename T>
	void addObject(T* object) {
		funcs.push_back({ &invokeFunc<T>, static_cast<void*>(object) });
	}

	void destroy() {
		for (auto& f : funcs) {
			f.func(f.ptr);
		}
		funcs.clear();
	}

	Init() {};
    Init(vkb::Instance instance, vkb::InstanceDispatchTable inst_disp, vkb::Device device, vkb::DispatchTable disp) :
        instance(instance), inst_disp(inst_disp), device(device), disp(disp) {
	}
    ~Init() {
        //destroy();
    }
};

uint32_t get_memory_index(Init& init, const uint32_t type_bits, VkMemoryPropertyFlags properties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) {
    VkPhysicalDeviceMemoryProperties mem_props = init.device.physical_device.memory_properties;
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        if ((type_bits & (1 << i)) && (mem_props.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    return UINT32_MAX; // No valid memory type found
}

struct Allocator {
    
    Allocator(Init* init) : init(init) {
        commandBuffers.resize(1);
        get_queues();
		create_command_pool();
    };

    Allocator() {};

    ~Allocator() {
        if (!commandBuffers.empty()) {
            init->disp.freeCommandBuffers(commandPool, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());
			commandBuffers.clear();
        }
		
        for(auto& [buf, mem] : allocated) {
            killMemory(buf, mem);
        }

        for (auto& [im, mem] : images) {
            killImage(im, mem);
        }

		init->disp.destroyCommandPool(commandPool, nullptr);
    }
    
    size_t getAlignmemt() const {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(init->device.physical_device, &props);
		return props.limits.minStorageBufferOffsetAlignment;
	}
    
    Allocator(Allocator& other) {
        init = other.init;
        allocQueue = other.allocQueue;
        commandPool = other.commandPool;
        commandBuffers = other.commandBuffers;
    }

    void killMemory(VkBuffer buffer, VkDeviceMemory memory) const {
        init->disp.destroyBuffer(buffer, nullptr);
        init->disp.freeMemory(memory, nullptr);
    }

    void killImage(VkImage image, VkDeviceMemory memory) const {
        init->disp.destroyImage(image, nullptr);
        init->disp.freeMemory(memory, nullptr);
    }

    std::pair<VkBuffer, VkDeviceMemory> createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, bool usingDescriptors = true) {
        VkBuffer buffer{};
		VkDeviceMemory bufferMemory{};

        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        init->disp.createBuffer(&bufferInfo, nullptr, &buffer);

        VkMemoryRequirements memRequirements;
        init->disp.getBufferMemoryRequirements(buffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = get_memory_index(*init, memRequirements.memoryTypeBits, properties);
        
        VkMemoryAllocateFlagsInfo allocFlagsInfo{};
        allocFlagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
        allocFlagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

		allocInfo.pNext = &allocFlagsInfo;

        if (init->disp.allocateMemory(&allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
            throw std::runtime_error("could not allocate memory");
        }
        if (init->disp.bindBufferMemory(buffer, bufferMemory, 0) != VK_SUCCESS) {
            throw std::runtime_error("could not bind memory");
        }
        allocated.emplace_back(buffer, bufferMemory);
        return { buffer, bufferMemory };
    };

	std::pair<VkImage, VkDeviceMemory> createImage(VkDeviceSize width, VkDeviceSize height, uint32_t mipLevels, VkImageUsageFlags usage, VkImageType imageType, VkMemoryPropertyFlags properties, VkFormat format = VK_FORMAT_R8G8B8A8_UNORM,VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT, VkSharingMode sharingMode = VK_SHARING_MODE_EXCLUSIVE) {
		VkImage image{};
		VkDeviceMemory imageMemory{};
		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = imageType;
		imageInfo.extent.width = static_cast<uint32_t>(width);
		imageInfo.extent.height = static_cast<uint32_t>(height);
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = mipLevels;
		imageInfo.arrayLayers = 1;
		imageInfo.format = format;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		imageInfo.usage = usage;
		imageInfo.samples = samples;
		imageInfo.sharingMode = sharingMode;
		init->disp.createImage(&imageInfo, nullptr, &image);
		VkMemoryRequirements memRequirements;
		init->disp.getImageMemoryRequirements(image, &memRequirements);
		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = get_memory_index(*init, memRequirements.memoryTypeBits, properties);
		if (init->disp.allocateMemory(&allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
			throw std::runtime_error("could not allocate memory");
		}
		if (init->disp.bindImageMemory(image, imageMemory, 0) != VK_SUCCESS) {
			throw std::runtime_error("could not bind memory");
		}
        images.emplace_back(image, imageMemory);
		return { image, imageMemory };
	}

    void fillBuffer(VkBuffer buffer, VkDeviceMemory memory, uint32_t data, VkDeviceSize offset = 0, VkDeviceSize range = VK_WHOLE_SIZE) {
		auto cmd = beginSingleTimeCommands();
        vkCmdFillBuffer(commandBuffers[cmd], buffer, offset, range, data);
		endSingleTimeCommands(true, false);
    }

	// avoid as much as possible, this kills performance. But if you need to free memory, this is the way to do it.
    void freeMemory(VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize range) {  
        // Step 1: Create a staging buffer to temporarily hold the data
        VkDeviceSize bufferSize;

        // Get memory requirements for the buffer
        VkMemoryRequirements memRequirements;
        init->disp.getBufferMemoryRequirements(buffer, &memRequirements);
        bufferSize = memRequirements.size;

        // Create a staging buffer
        auto [stagingBuffer, stagingMemory] = createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        // Step 2: Copy data before the gap to the staging buffer
        if (offset > 0) {
            sequentialCopyBuffer(buffer, stagingBuffer, offset, 0, 0);
        }

        // Step 3: Copy data after the gap to the staging buffer
        VkDeviceSize end = offset + range;
        VkDeviceSize remainingSize = (end <= bufferSize) ? bufferSize - end : 0;
        if (remainingSize > 0) {
            sequentialCopyBuffer(buffer, stagingBuffer, remainingSize, offset + range, offset);
        }

        // Step 4: Copy the data back from the staging buffer to the original buffer
        if (offset > 0) {
            sequentialCopyBuffer(stagingBuffer, buffer, offset, 0, 0);
        }

        if (remainingSize > 0) {  
            sequentialCopyBuffer(stagingBuffer, buffer, remainingSize, offset, offset);
        }

        submitAllCommands(true);

        // Step 5: Clear the staging buffer
        init->disp.destroyBuffer(stagingBuffer, nullptr);
        init->disp.freeMemory(stagingMemory, nullptr);
    }

	// this overrides the memory in offset -> insertSize with toInsert(0 -> insertSize).
    void replaceMemory(VkBuffer& buffer, VkDeviceMemory& memory, VkBuffer toInsert, VkDeviceSize insertSize, VkDeviceSize offset) {
        // Step 1: Get memory requirements
        VkMemoryRequirements bufferReq;
        init->disp.getBufferMemoryRequirements(buffer, &bufferReq);

        VkDeviceSize totalSize = bufferReq.size;

        // Step 2: Create staging buffer
        auto [stagingBuffer, stagingMemory] = createBuffer(totalSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        // Step 3: Copy data before insertion point
        if (offset > 0) {
            sequentialCopyBuffer(buffer, stagingBuffer, offset, 0, 0);
        }

        // Step 4: Copy new data to insert
        sequentialCopyBuffer(toInsert, stagingBuffer, insertSize, 0, offset);

        // Step 5: Copy remaining original buffer data
        VkDeviceSize remainingSize = totalSize - (offset + insertSize);
        if (remainingSize > 0) {
            sequentialCopyBuffer(buffer, stagingBuffer, remainingSize, offset, offset + insertSize);
        }

		sequentialCopyBuffer(stagingBuffer, buffer, totalSize, 0, 0);

        submitAllCommands(true);

        // Step 6: Destroy staging buffer
        init->disp.destroyBuffer(stagingBuffer, nullptr);
        init->disp.freeMemory(stagingMemory, nullptr);
    }

    // insert blob of memory into given buffer. Make sure given buffer has enough free space to handle the insert block, otherwise it'll kill the remaining trailing data
    void insertMemory(VkBuffer buffer, VkDeviceMemory memory, VkBuffer& toInsert, VkDeviceSize offset, VkDeviceSize inSize) {
        VkMemoryRequirements req;
        init->disp.getBufferMemoryRequirements(buffer, &req);
        auto size = req.size;

        auto [stagingBuffer, stagingMemory] = createBuffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        if (offset > 0) {
            sequentialCopyBuffer(buffer, stagingBuffer, offset, 0, 0);
        }
        sequentialCopyBuffer(toInsert, stagingBuffer, inSize, 0, offset);

        sequentialCopyBuffer(buffer, stagingBuffer, size - (offset + inSize), offset, offset + inSize);

        sequentialCopyBuffer(stagingBuffer, buffer, size, 0, 0);

        submitAllCommands(true);

        init->disp.destroyBuffer(stagingBuffer, nullptr);
        init->disp.freeMemory(stagingMemory, nullptr);
    }

	// this is a defragmenter. It will copy the good data from the original buffer to a new buffer, and then free the original buffer, killing the stale memory
    void defragment(VkBuffer& buffer, VkDeviceMemory& memory, std::vector<std::pair<VkDeviceSize, VkDeviceSize>>& aliveMem) {
		// create new staging buffer which will replace the original buffer
		VkDeviceSize bufferSize;
		// Get memory requirements for the buffer
		VkMemoryRequirements memRequirements;
		init->disp.getBufferMemoryRequirements(buffer, &memRequirements);
		bufferSize = memRequirements.size;
		// Create the clean buffer
		auto [stagingBuffer, stagingMemory] = createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        VkDeviceSize runningOffset = 0;
        // copy all the good buffers into the staging buffer, one after the other
		for (auto& [offset, range] : aliveMem) {
			// copy the data from the original buffer to the new one
			sequentialCopyBuffer(buffer, stagingBuffer, range, offset, runningOffset);
			runningOffset += range;
		}

		// copy the curated data from the staging buffer to the original buffer
		sequentialCopyBuffer(stagingBuffer, buffer, bufferSize, 0, 0);

		submitAllCommands(true);

		init->disp.destroyBuffer(stagingBuffer, nullptr);
		init->disp.freeMemory(stagingMemory, nullptr);
    }

    void transitionImageLayout(
        VkImage image,
        VkFormat format,
        VkImageLayout oldLayout,
        VkImageLayout newLayout,
        uint32_t mipLevels = 1) {
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;

        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

        barrier.image = image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = mipLevels;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;

        // Determine source and destination stages
        VkPipelineStageFlags sourceStage, destinationStage;

        if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
            newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

            sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        }
        else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
            newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }
        else {
            throw std::invalid_argument("unsupported layout transition!");
        }
        auto cmd = beginSingleTimeCommands(true);
        vkCmdPipelineBarrier(
            commandBuffers[cmd],
            sourceStage, destinationStage,
            0,
            0, nullptr,
            0, nullptr,
            1, &barrier
        );
        endSingleTimeCommands(true, true, true);
    }

    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size, VkDeviceSize srcOffset, VkDeviceSize dstOffset, bool async = false) {
        auto cmd = beginSingleTimeCommands();
        VkBufferCopy copyRegion{};
        copyRegion.size = size;
        copyRegion.dstOffset = dstOffset;
        copyRegion.srcOffset = srcOffset;
        vkCmdCopyBuffer(commandBuffers[cmd], srcBuffer, dstBuffer, 1, &copyRegion);
        endSingleTimeCommands(async);
    }

	void sequentialCopyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size, VkDeviceSize srcOffset, VkDeviceSize dstOffset) {
        auto cmd = beginSingleTimeCommands();
		VkBufferCopy copyRegion{};
		copyRegion.size = size;
		copyRegion.dstOffset = dstOffset;
		copyRegion.srcOffset = srcOffset;
		vkCmdCopyBuffer(commandBuffers[cmd], srcBuffer, dstBuffer, 1, &copyRegion);
        endSingleTimeCommands(true, false);
	}

    // only supports sqare images
	void copyImage(VkImage srcImage, VkImage dstImage, VkDeviceSize size, VkDeviceSize srcOffset, VkDeviceSize dstOffset) {
		auto cmd = beginSingleTimeCommands();
		VkImageCopy copyRegion{};
		copyRegion.extent.width = static_cast<uint32_t>(size);
		copyRegion.extent.height = static_cast<uint32_t>(size);
		copyRegion.extent.depth = 1;
		copyRegion.dstOffset.x = static_cast<int32_t>(dstOffset);
		copyRegion.dstOffset.y = static_cast<int32_t>(dstOffset);
		copyRegion.srcOffset.x = static_cast<int32_t>(srcOffset);
		copyRegion.srcOffset.y = static_cast<int32_t>(srcOffset);
		vkCmdCopyImage(commandBuffers[cmd], srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dstImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);
		endSingleTimeCommands();
	}

    // does not support mipmaps or array textures
    void copyBufferToImage2D(VkBuffer srcBuffer, VkImage dstImage, uint32_t width, uint32_t height) {
        auto cmd = beginSingleTimeCommands();

        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;

        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;

        region.imageOffset = { 0, 0, 0 };
        region.imageExtent = {
            width,
            height,
            1
        };

        vkCmdCopyBufferToImage(
            commandBuffers[cmd],
            srcBuffer,
            dstImage,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &region
        );

		endSingleTimeCommands(true, true, false);
    }

	// this begins the command buffer recording process. Just record the commands in between this function's call and the submitSingleTimeCmd call
	// submitSingleTimeCmd will NOT automatically end the command buffer!!
	VkCommandBuffer getSingleTimeCmd(bool useGraphicsQueue = false) const {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        if (!useGraphicsQueue) {
            allocInfo.commandPool = commandPool;
        }
        else {
            allocInfo.commandPool = graphicsPool;
        }

        allocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer;
        init->disp.allocateCommandBuffers(&allocInfo, &commandBuffer);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        init->disp.beginCommandBuffer(commandBuffer, &beginInfo);
		return commandBuffer;
	}

	void submitSingleTimeCmd(VkCommandBuffer cmd, bool async = true, bool useGraphicsQueue = false) {
		SubmitSingleTimeCommand(cmd, async, useGraphicsQueue);
	}

    Init* init;
    VkCommandPool commandPool;
    VkCommandPool graphicsPool;
    VkQueue graphicsQueue;
private:
    VkQueue allocQueue;
    std::vector<VkCommandBuffer> commandBuffers;
    std::vector<std::pair<VkBuffer, VkDeviceMemory>> allocated;
    std::vector<std::pair<VkImage, VkDeviceMemory>> images;

    int beginSingleTimeCommands(bool useGraphicsQueue = false) {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        if (!useGraphicsQueue) {
            allocInfo.commandPool = commandPool;
		}
		else {
			allocInfo.commandPool = graphicsPool;
		}
        
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer;
        init->disp.allocateCommandBuffers(&allocInfo, &commandBuffer);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        init->disp.beginCommandBuffer(commandBuffer, &beginInfo);
        commandBuffers.push_back(commandBuffer);
		return commandBuffers.size() - 1; // return the index at which we just pushed the command buffer
    }

    int SubmitSingleTimeCommand(VkCommandBuffer commandBuffer, bool async = true, bool useGraphicsQueue = false) const {
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;
        if (async) {
            if (useGraphicsQueue) {
			    init->disp.queueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
                init->disp.queueWaitIdle(graphicsQueue);
                init->disp.freeCommandBuffers(graphicsPool, 1, &commandBuffer);
		    }
		    else if (!useGraphicsQueue) {
			    init->disp.queueSubmit(allocQueue, 1, &submitInfo, VK_NULL_HANDLE);
                init->disp.queueWaitIdle(allocQueue);
				init->disp.freeCommandBuffers(commandPool, 1, &commandBuffer);
		    }
        }
        else {
            if (useGraphicsQueue) {
                init->disp.queueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
            }
            else if (!useGraphicsQueue) {
                init->disp.queueSubmit(allocQueue, 1, &submitInfo, VK_NULL_HANDLE);
            }
            init->disp.deviceWaitIdle();

            if (useGraphicsQueue) {
			    init->disp.freeCommandBuffers(graphicsPool, 1, &commandBuffer);
		    }
		    else {
			    init->disp.freeCommandBuffers(commandPool, 1, &commandBuffer);
		    }
        }
		return 0;
    }

    void endSingleTimeCommands(bool async = false, bool dispatch = true, bool dispathOnGraphics = false) {
        if (commandBuffers.empty()) return;

        auto cmd = commandBuffers.back();
        init->disp.endCommandBuffer(cmd);

        if (dispatch) {
            VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &cmd;
			if (dispathOnGraphics) {
				init->disp.queueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
			}
			else {
				init->disp.queueSubmit(allocQueue, 1, &submitInfo, VK_NULL_HANDLE);
			}

            if (!async) {
                init->disp.deviceWaitIdle();
            }
            else {
                if (!dispathOnGraphics) {
                    init->disp.queueWaitIdle(allocQueue);
				}
                else {
                    init->disp.queueWaitIdle(graphicsQueue);
                }
            }
            if (dispathOnGraphics) {
                init->disp.freeCommandBuffers(graphicsPool, 1, &cmd);
            }
            else {
                init->disp.freeCommandBuffers(commandPool, 1, &cmd);
            }
            commandBuffers.pop_back();
        }
    }

    void submitAllCommands(bool async = false) {
        if (commandBuffers.empty()) return;
		//commandBuffers.erase(commandBuffers.begin()); // the first command buffer is apparently invalid?! will have to fix later, right now, I'm exhausted.
        VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = commandBuffers.size();
            submitInfo.pCommandBuffers = commandBuffers.data();
            init->disp.queueSubmit(allocQueue, 1, &submitInfo, VK_NULL_HANDLE);
        if (async) { init->disp.queueWaitIdle(allocQueue); }    // this stalls the transfer queue, but not the whole device.
		else { init->disp.deviceWaitIdle(); } 		// this stalls the whole device. This is terrible. But it's useful for stuff where you need to wait for the GPU to finish before you can do anything else.
		
        init->disp.freeCommandBuffers(commandPool, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());
        commandBuffers.clear();
    }

    int get_queues() {
        auto gq = init->device.get_queue(vkb::QueueType::transfer);
        if (!gq.has_value()) {
            std::cout << "failed to get queue: " << gq.error().message() << "\n";
            return -1;
        }
        allocQueue = gq.value();
        return 0;
    }

    void create_command_pool() {
        VkCommandPoolCreateInfo pool_info = {};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.queueFamilyIndex = init->device.get_queue_index(vkb::QueueType::transfer).value();
        init->disp.createCommandPool(&pool_info, nullptr, &commandPool);

        VkCommandBufferAllocateInfo allocate_info = {};
        allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocate_info.commandPool = commandPool;
        allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocate_info.commandBufferCount = commandBuffers.size();
        //init->disp.allocateCommandBuffers(&allocate_info, commandBuffers.data());
    }
};

// lone wolf buffer. Has its own VkDeviceMemory. Uses a staging buffer to copy memory from host to GPU high-performance memory
template<typename T>
struct StandaloneBuffer {
    VkBuffer buffer{};
    VkDeviceMemory bufferMemory{};

	VkDeviceAddress bufferAddress{}; // Address of the buffer in GPU memory.

    VkBuffer stagingBuffer{};
	VkDeviceMemory stagingBufferMemory{}; // Staging memory on CPU (same size as buffer and memory on GPU)

    VkDeviceSize alignment;
    VkDeviceSize capacity;
	uint32_t numElements = 0; // Number of elements in this buffer

    // stuff you need to send to pipeline creation:
    VkDescriptorSetLayoutBinding binding{};
    VkWriteDescriptorSet wrt_desc_set{};
    VkDescriptorBufferInfo desc_buf_info{};
    uint32_t bindingIndex;

	// for transfer operations to be done on seperate queue
	Allocator* allocator;

	VkShaderStageFlagBits flags = VK_SHADER_STAGE_COMPUTE_BIT;

    void* memMap;

	// Signal to all resources using this buffer. The current descriptor set is invalid and needs to be updated and this varieable needs to be set to false
	Signal<10> descUpdateQueued;
    
    // Copy assignment operator
    StandaloneBuffer& operator=(const StandaloneBuffer& other) {
        if (this != &other) {
            buffer = other.buffer;
            bufferMemory = other.bufferMemory;
            stagingBuffer = other.stagingBuffer;
            stagingBufferMemory = other.stagingBufferMemory;
            alignment = other.alignment;
            capacity = other.capacity;
            numElements = other.numElements;
            binding = other.binding;
            wrt_desc_set = other.wrt_desc_set;
            desc_buf_info = other.desc_buf_info;
            bindingIndex = other.bindingIndex;
            allocator = other.allocator;
            flags = other.flags;
            memMap = other.memMap;
            descUpdateQueued = other.descUpdateQueued;
        }
        return *this;
    }

    // Copy constructor
    StandaloneBuffer(const StandaloneBuffer& other)
        : buffer(other.buffer),
        bufferMemory(other.bufferMemory),
        stagingBuffer(other.stagingBuffer),
        stagingBufferMemory(other.stagingBufferMemory),
        alignment(other.alignment),
        capacity(other.capacity),
        numElements(other.numElements),
        binding(other.binding),
        wrt_desc_set(other.wrt_desc_set),
        desc_buf_info(other.desc_buf_info),
        bindingIndex(other.bindingIndex),
        allocator(other.allocator),
        flags(other.flags),
        memMap(other.memMap),
        descUpdateQueued(other.descUpdateQueued),
        bufferAddress(other.bufferAddress){
    }

    StandaloneBuffer(size_t numElements, Allocator* allocator, VkShaderStageFlagBits flags = VK_SHADER_STAGE_COMPUTE_BIT) : allocator(allocator), flags(flags), numElements(numElements) {
        init();
    }

    StandaloneBuffer(std::vector<T>& data, Allocator* allocator, VkShaderStageFlagBits flags = VK_SHADER_STAGE_COMPUTE_BIT) : allocator(allocator), flags(flags) {
		numElements = static_cast<uint32_t>(data.size());
        init();
        alloc(data);
    }

    StandaloneBuffer() : allocator(nullptr) {}

    void init() {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(allocator->init->device.physical_device, &props);
        alignment = props.limits.minStorageBufferOffsetAlignment;

        capacity = numElements * sizeof(T);
        capacity = (capacity + alignment - 1) & ~(alignment - 1);

        // Create the staging buffer
        auto stageBuff = allocator->createBuffer(capacity, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, true);
        stagingBuffer = stageBuff.first;
        stagingBufferMemory = stageBuff.second;

        // Create the buffer
        auto buff = allocator->createBuffer(capacity, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, true);
        buffer = buff.first;
        bufferMemory = buff.second;
        allocator->init->disp.mapMemory(stagingBufferMemory, 0, capacity, 0, &memMap);

		VkBufferDeviceAddressInfoEXT bufferInfo{};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_EXT;
		bufferInfo.buffer = buffer;
		bufferInfo.pNext = nullptr;

		bufferAddress = allocator->init->disp.getBufferDeviceAddress(&bufferInfo);
        std::cout << "wefwef\n";
    }

	VkDeviceAddress getBufferAddress() {
        VkBufferDeviceAddressInfoEXT bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_EXT;
        bufferInfo.buffer = buffer;
        bufferInfo.pNext = nullptr;

        bufferAddress = allocator->init->disp.getBufferDeviceAddress(&bufferInfo);
        return bufferAddress;
	}

    void createDescriptors(int idx, VkShaderStageFlagBits stage) {
        binding.binding = stage;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binding.descriptorCount = 1;
        binding.stageFlags = stage;
        binding.pImmutableSamplers = nullptr;
    }

    void createDescriptors() {
        binding.binding = bindingIndex;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binding.descriptorCount = 1;
        binding.stageFlags = flags;
        binding.pImmutableSamplers = nullptr;
    }

    void updateDescriptorSet(VkDescriptorSet& set) {
        desc_buf_info.buffer = buffer;
        desc_buf_info.offset = 0;
        desc_buf_info.range = VK_WHOLE_SIZE;

        wrt_desc_set.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wrt_desc_set.dstSet = set;
        wrt_desc_set.dstBinding = bindingIndex;
        wrt_desc_set.dstArrayElement = 0;
        wrt_desc_set.descriptorCount = 1;
        wrt_desc_set.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wrt_desc_set.pBufferInfo = &desc_buf_info;
    }

    void updateDescriptorSet(VkDescriptorSet& set, int arrayElement, int idx) {
        desc_buf_info.buffer = buffer;
        desc_buf_info.offset = 0;
        desc_buf_info.range = VK_WHOLE_SIZE;

        wrt_desc_set.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wrt_desc_set.dstSet = set;
        wrt_desc_set.dstBinding = idx;
        wrt_desc_set.dstArrayElement = arrayElement;
        wrt_desc_set.descriptorCount = 1;
        wrt_desc_set.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wrt_desc_set.pBufferInfo = &desc_buf_info;
    }

    void clearBuffer() {
        if (allocator != nullptr) {
            allocator->freeMemory(buffer, bufferMemory, 0, capacity);
            allocator->freeMemory(stagingBuffer, stagingBufferMemory, 0, capacity);
        }
    }

	// Allocate data into the buffer. Each alloc will overwrite the previous data in the buffer.
    void alloc(std::vector<T>& data) {
        auto sizeOfData = static_cast<uint32_t>(sizeof(T) * data.size());
        std::memcpy(memMap, data.data(), sizeOfData);
        allocator->copyBuffer(stagingBuffer, buffer, sizeOfData, 0, 0, true);
		std::memset(memMap, 0, sizeOfData);
		numElements = static_cast<uint32_t>(data.size());
    }

    void grow(int factor) {
		auto oldCapacity = capacity;
        capacity = factor * sizeof(T);
		capacity = (capacity + alignment - 1) & ~(alignment - 1);

		auto [buf, mem] = allocator->createBuffer(capacity, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, true);

		allocator->copyBuffer(buffer, buf, oldCapacity, 0, 0, true);

        allocator->init->disp.destroyBuffer(buffer, nullptr);
		allocator->init->disp.freeMemory(bufferMemory, nullptr);
		buffer = buf;
		bufferMemory = mem;

		getBufferAddress();

        // the internal handles were changed. We need descriptor updates
        descUpdateQueued.trigger();
    }

    // newSize is size of buffer in elements, NOT bytes
    void resize(uint32_t newSize) {
        auto prevCapacity = capacity;
		capacity = newSize * sizeof(T);
		capacity = (capacity + alignment - 1) & ~(alignment - 1);

		auto [buf, mem] = allocator->createBuffer(capacity, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, true);
		allocator->copyBuffer(buffer, buf, prevCapacity, 0, 0, true);
		allocator->init->disp.destroyBuffer(buffer, nullptr);
		allocator->init->disp.freeMemory(bufferMemory, nullptr);

		buffer = buf;
		bufferMemory = mem;
		getBufferAddress();
		// the internal handles were changed. We need descriptor updates
		descUpdateQueued.trigger();
    }

    size_t size() const {
        return capacity / sizeof(T);
    }

    std::vector<T> downloadBuffer() {
        
		VkMemoryRequirements memRequirements;
		allocator->init->disp.getBufferMemoryRequirements(buffer, &memRequirements);
		auto bufferSize = memRequirements.size;
		auto numElements = static_cast<uint32_t>(bufferSize / sizeof(T));
        allocator->copyBuffer(buffer, stagingBuffer, bufferSize, 0, 0, true);

		std::vector<T> data; 
        data.resize(numElements);
        std::memcpy(data.data(), memMap, bufferSize);
		std::memset(memMap, 0, bufferSize);
		return data;
    }

    ~StandaloneBuffer() {
        allocator->init->disp.unmapMemory(stagingBufferMemory);
    };
};

// Bindless descriptor array of standaloneBuffers
template<typename bufferType>
struct BufferArray {
    std::vector<StandaloneBuffer<bufferType>> buffers;

    uint32_t numBuffers = 1000;             // by default, 1000 buffers are supported

	VkDescriptorSet descSet;
	VkDescriptorSetLayout descSetLayout;
	VkDescriptorPool descPool;
	Allocator* allocator;

    uint32_t bindingIndex = 0;

	BufferArray(Allocator* allocator, uint32_t bindingIndex) : allocator(allocator), bindingIndex(bindingIndex) {
        createDescriptorPool();
		createDescSetLayout();
		allocateDescSet();
        allocator->init->addObject(this);
	}

    BufferArray(Allocator* allocator, uint32_t bindingIndex, VkDescriptorPool sharedPool) : allocator(allocator), bindingIndex(bindingIndex), descPool(sharedPool) {
        createDescSetLayout();
		allocateDescSet();
    };

    BufferArray() {};

    void destructor() {
		allocator->init->disp.destroyDescriptorPool(descPool, nullptr);
		allocator->init->disp.destroyDescriptorSetLayout(descSetLayout, nullptr);
    }

	void push_back(StandaloneBuffer<bufferType>& buffer) {
		if (buffers.size() >= numBuffers) {
			std::cout << "BufferArray: reached max number of buffers. Cannot add more.\n";
			return;
		}
		buffers.push_back(std::move(buffer));
	}

	void push_back(std::vector<bufferType>& data) {
		if (buffers.size() >= numBuffers) {
			std::cout << "BufferArray: reached max number of buffers. Cannot add more.\n";
			return;
		}
		buffers.emplace_back(data, allocator);
	}

	void erase(uint32_t idx) {
		if (idx < 0 || idx >= buffers.size()) {
			std::cout << "BufferArray: index out of range. Cannot erase.\n";
			return;
		}
		allocator->init->disp.destroyBuffer(buffers[idx].buffer, nullptr);
		allocator->init->disp.freeMemory(buffers[idx].bufferMemory, nullptr);
		buffers.erase(buffers.begin() + idx);
	}

	size_t size() const {
		return buffers.size();
	}

    void createDescriptorPool() {
        VkDescriptorPoolSize pool_sizes_bindless[] =
        {
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, numBuffers }
        };

        VkDescriptorPoolCreateInfo poolInfo = {};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = pool_sizes_bindless;
        poolInfo.maxSets = 1;
        poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
        allocator->init->disp.createDescriptorPool(&poolInfo, nullptr, &descPool);
    }

    void createDescSetLayout() {
        VkDescriptorBindingFlags bindless_flags = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT_EXT | VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT_EXT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT_EXT;

        VkDescriptorSetLayoutBinding vk_binding;
        vk_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        vk_binding.descriptorCount = numBuffers;
        vk_binding.binding = 0;

        vk_binding.stageFlags = VK_SHADER_STAGE_ALL;
        vk_binding.pImmutableSamplers = nullptr;

        VkDescriptorSetLayoutCreateInfo layout_info = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        layout_info.bindingCount = 1;
        layout_info.pBindings = &vk_binding;
        layout_info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT_EXT;

        VkDescriptorSetLayoutBindingFlagsCreateInfoEXT extended_info{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO_EXT, nullptr };
        extended_info.bindingCount = 1;
        extended_info.pBindingFlags = &bindless_flags;

        layout_info.pNext = &extended_info;

        allocator->init->disp.createDescriptorSetLayout(&layout_info, nullptr, &descSetLayout);
    }

    void allocateDescSet() {
        VkDescriptorSetAllocateInfo alloc_info{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        alloc_info.descriptorPool = descPool;
        alloc_info.descriptorSetCount = 1;
        alloc_info.pSetLayouts = &descSetLayout;

        VkDescriptorSetVariableDescriptorCountAllocateInfoEXT count_info{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO_EXT };
        uint32_t max_binding = numBuffers - 1;
        count_info.descriptorSetCount = 1;
        // This number is the max allocatable count
        count_info.pDescriptorCounts = &max_binding;
        alloc_info.pNext = &count_info;

        allocator->init->disp.allocateDescriptorSets(&alloc_info, &descSet);
    }

    void updateDescriptorSets() {
        std::vector<VkWriteDescriptorSet> writes(buffers.size());
        for (size_t i = 0; i < buffers.size(); ++i) {
            buffers[i].createDescriptors(0, VK_SHADER_STAGE_ALL);
            buffers[i].updateDescriptorSet(descSet, i, bindingIndex);
            writes[i] = buffers[i].wrt_desc_set;
        }
        allocator->init->disp.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }
};

// an element inside the MemPool.
template<typename T>
struct Buffer {
    VkBuffer buffer;                      // Points to the MemPool's buffer
    VkDeviceSize offset;                  // Byte Offset within the buffer
	uint32_t elementOffset;               // Element Offset within the buffer
    uint32_t numElements;                 // Number of elements in this buffer
	VkDeviceSize alignedSize(VkDeviceSize alignment) { return (numElements * sizeof(T) + alignment - 1) & ~(alignment - 1); } // Return the aligned byte size of this buffer element
	VkDeviceSize size() { return numElements * sizeof(T); } // Return the byte size of this buffer element

    // Descriptor set members (used when MemPool is not using bypassDescriptors)
    uint32_t bindingIndex;
    VkDescriptorSetLayoutBinding binding{};
    VkWriteDescriptorSet wrt_desc_set{};
    VkDescriptorBufferInfo desc_buf_info{};

    void createDescriptors(uint32_t bindingIdx, VkShaderStageFlags flags= VK_SHADER_STAGE_COMPUTE_BIT) {
        bindingIndex = bindingIdx;
        binding.binding = bindingIndex;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binding.descriptorCount = 1;
        binding.stageFlags = flags;
        binding.pImmutableSamplers = nullptr;
    }

    void updateDescriptorSet(VkDescriptorSet& set) {
        desc_buf_info.buffer = buffer;
        desc_buf_info.offset = offset;
        desc_buf_info.range = numElements * sizeof(T);

        wrt_desc_set.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wrt_desc_set.dstSet = set;
        wrt_desc_set.dstBinding = bindingIndex;
        wrt_desc_set.dstArrayElement = 0;
        wrt_desc_set.descriptorCount = 1;
        wrt_desc_set.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wrt_desc_set.pBufferInfo = &desc_buf_info;
    }
};

// like std::vector, but worse. And for the GPU. Only handles Storage buffers. For Uniform buffers, use different struct
// Each input is a seperate descriptor binding, but same set.
// Can update buffer offsets and even buffer handles if defragmented or resized. Shoots a Signal struct to all using resources. Resources using this pool must register themselves as a 
// signal listener and submit their onSignal() function ptr to the pool's Signal. When internal descriptors are updated, you need to pause your pipeline's execution and update descriptors
// This is useful for dynamic memory allocation where all scene mesh data is allocated together, in one MemPool (won't work now cause we can't have custom buffer usage here yet, 
// only STORAGE_BUFFER is supported). You shouldn't need to use any defragmentation or resize operations if you're using this for training Machine Learning tensors.
template<typename T>
struct MemPool {
    VkBuffer buffer;               // Single buffer for all allocations on GPU
    VkDeviceMemory memory;         // Backing memory on GPU
	VkDeviceAddress poolAddress;   // Address of the buffer in GPU memory.

    // persistent staging buffer for efficiency. We should avoid allocating new memory whenever we can.
    // Not really the best solution, cause for every MemPool, 
    // we now have two memories of the same size that's taking up space in the computer... but eeh
    VkBuffer stagingBuffer;                  // Staging buffer on CPU (same size as buffer and memory on GPU)
    VkDeviceMemory stagingMemory;            // Staging memory on CPU (same size as buffer and memory on GPU)

    VkDeviceAddress bufferDeviceAddress; // Actual pointer to the memory inside of the GPU. used when bypassDescriptors are activated

    void* mapped = nullptr;        // Mapped pointer to the staging buffer. This stays mapped until MemPool is destroyed.

    VkDeviceSize alignment;        // Buffer alignment requirement
    VkDeviceSize capacity;         // Total capacity in bytes
    VkDeviceSize offset = 0;       // Current allocation byte offset
	uint32_t elementOffset = 0;    // Current element offset (in number of elements)
    VkDeviceSize occupied = 0;     // The amount of occupied memory in the buffer (its value of the same as current allocation offset, but I kept it seperate for readability purposes)

    std::vector<Buffer<T>> buffers; // Track all allocated buffers
    VkDeviceSize deadMemory = 0;

    Allocator* allocator;           // The allocator has all the vulkan context stuff and it handles buffer creation, destruction, manipulation, etc.

    VkShaderStageFlagBits flags = VK_SHADER_STAGE_COMPUTE_BIT;

    Signal<10> descUpdateQueued;

    // maxElements means number of individual elements of type T inside of all the buffers. 
    // Meaning, if maxElements is set to 10, pool can sustain 10 buffers with 1 float each (if T = float).
    // But only 2 buffers with 5 floats each.
    MemPool(uint32_t maxElements, Allocator* allocator = nullptr, VkShaderStageFlagBits flags = VK_SHADER_STAGE_COMPUTE_BIT) : allocator(allocator), flags(flags) {
        // Query alignment requirement for storage buffers
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(allocator->init->device.physical_device, &props);
        alignment = props.limits.minStorageBufferOffsetAlignment;

        if (maxElements == 0) {
            std::cerr << "maxElements cannot be 0" << std::endl;
            return;
        }

        // Calculate total size with alignment
        capacity = maxElements * sizeof(T);
        capacity = (capacity + alignment - 1) & ~(alignment - 1);

        // Create CPU staging buffer (CPU visible, low-performance)
        auto stageBuff = allocator->createBuffer(capacity, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        stagingBuffer = stageBuff.first;
        stagingMemory = stageBuff.second;

        // Map the staging buffer memory
        allocator->init->disp.mapMemory(stagingMemory, 0, capacity, 0, &mapped);

        auto buff = allocator->createBuffer(capacity,
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        buffer = buff.first;
        memory = buff.second;

		VkBufferDeviceAddressInfoEXT bufferInfo{};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_EXT;
		bufferInfo.buffer = buffer;
		bufferInfo.pNext = nullptr;
		poolAddress = allocator->init->disp.getBufferDeviceAddress(&bufferInfo);
    }

    MemPool() {};

	

    ~MemPool() {
        allocator->init->disp.unmapMemory(stagingMemory);
    }

    size_t size() {
        return buffers.size();
    }
    
	VkDeviceAddress getBufferAddress() {
		VkBufferDeviceAddressInfoEXT bufferInfo{};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_EXT;
		bufferInfo.buffer = buffer;
		bufferInfo.pNext = nullptr;
		poolAddress = allocator->init->disp.getBufferDeviceAddress(&bufferInfo);
		return poolAddress;
	}

    // simple push_back operation. Quite expensive, use with caution.
    bool push_back(const std::vector<T> data, bool autoBind = true) {
        auto bindingIndex = buffers.size();
        const VkDeviceSize dataSize = data.size() * sizeof(T);
        const VkDeviceSize alignedSize = (dataSize + alignment - 1) & ~(alignment - 1);
        occupied += alignedSize;
        // Check if there's enough space
        if (offset + alignedSize > capacity) {
            growUntil(2, offset + alignedSize);
        }

        VkMemoryRequirements memRequirements{};
        allocator->init->disp.getBufferMemoryRequirements(stagingBuffer, &memRequirements);
		auto stagingBufferSize = memRequirements.size;

        // Step 1: Copy Data to Staging Buffer
		if (stagingBufferSize < alignedSize) {
            allocator->init->disp.unmapMemory(stagingMemory);
			allocator->init->disp.destroyBuffer(stagingBuffer, nullptr);
			allocator->init->disp.freeMemory(stagingMemory, nullptr);
			auto stageBuff = allocator->createBuffer(alignedSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
			stagingBuffer = stageBuff.first;
			stagingMemory = stageBuff.second;
			allocator->init->disp.mapMemory(stagingMemory, 0, alignedSize, 0, &mapped);
		}
        std::memcpy(mapped, data.data(), dataSize);

        // Step 2: Copy Staging Buffer to GPU Buffer
        allocator->copyBuffer(stagingBuffer, buffer, alignedSize, 0, offset, true);

        // Clear the staging buffer memory
        std::memset(mapped, 0, alignedSize);

        // Create a Buffer entry
        Buffer<T> newBuffer;
        newBuffer.buffer = buffer;
        newBuffer.offset = offset;
		newBuffer.elementOffset = elementOffset;
        newBuffer.numElements = static_cast<uint32_t>(data.size());
        newBuffer.createDescriptors(bindingIndex, flags);
        buffers.push_back(newBuffer);

        // Update offset for next allocation
        offset += alignedSize;
		elementOffset += static_cast<uint32_t>(data.size());
        return true;
    }

    // the standaloneBuffer still lives after this. If you want to dispose of it, you'll have to call its destructor manually
    bool push_back(StandaloneBuffer<T>& data) {
        auto bindingIndex = buffers.size();

        // Check if there's enough space
        if (offset + data.capacity > capacity) {
            grow(2);
        }

        // Copy given buffer to GPU Buffer
        allocator->copyBuffer(data.buffer, buffer, data.capacity, 0, offset, true);

        // Create a Buffer entry
        Buffer<T> newBuffer;
        newBuffer.buffer = buffer;
        newBuffer.offset = offset;
        newBuffer.numElements = static_cast<uint32_t>(data.capacity / sizeof(T));
        newBuffer.createDescriptors(bindingIndex, flags);
        buffers.push_back(newBuffer);
        // Update offset for next allocation
        offset += data.capacity;
        elementOffset += data.numElements;
        occupied += data.capacity;
        return true;
    }

    // Retrieve data from GPU
    std::vector<T> operator[](size_t index) {
        if (index >= buffers.size() || index < 0) {
            throw std::out_of_range("Index out of range");
        }

        Buffer<T>& buf = buffers[index];
        VkDeviceSize dataSize = buf.numElements * sizeof(T);
        const VkDeviceSize alignedSize = (dataSize + alignment - 1) & ~(alignment - 1);

        // Copy Data from GPU Buffer to Staging Buffer
        allocator->copyBuffer(buffer, stagingBuffer, dataSize, buf.offset, 0);

        std::vector<T> output(buf.numElements);
        std::memcpy(output.data(), mapped, dataSize);
        std::memset(mapped, 0, alignedSize);

        return output;
    }

    // resizes the memPool to the given newSize. This doesn't destroy data in the original buffer.
    void resize(int newSize) {
        // Calculate total size with alignment
        capacity = newSize * sizeof(T);
        capacity = (capacity + alignment - 1) & ~(alignment - 1);

        auto newBuffer = allocator->createBuffer(capacity,
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        allocator->copyBuffer(buffer, newBuffer.first, capacity, 0, 0, true);

        allocator->init->disp.destroyBuffer(buffer, nullptr);
        allocator->init->disp.freeMemory(memory, nullptr);

        buffer = newBuffer.first;
        memory = newBuffer.second;

        for (auto& buffer : buffers) {
            buffer.buffer = newBuffer.first;
        }
		getBufferAddress();
        // Internal handles were changed. We need descriptor updates
        descUpdateQueued.trigger();
    }

    // Grow the buffer to a new size. This will not change the data inside of the buffer.
    // Very expensive Operation. Try to set the MemPool's size at the beginning of it's life, that way there's no need for growth.
    void grow(int factor) {
        // Calculate total size with alignment
        auto oldCapacity = capacity;

        capacity *= factor;
        capacity = (capacity + alignment - 1) & ~(alignment - 1);

        // Notice how we don't grow the staging buffer. This can cause problems when trying to download the buffer to host, but eeh
        auto newBuffer = allocator->createBuffer(capacity,
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        allocator->copyBuffer(buffer, newBuffer.first, oldCapacity, 0, 0, true);

        allocator->init->disp.destroyBuffer(buffer, nullptr);
        allocator->init->disp.freeMemory(memory, nullptr);

        buffer = newBuffer.first;
        memory = newBuffer.second;

        for (auto& buffer : buffers) {
            buffer.buffer = newBuffer.first;
        }
        getBufferAddress();

        // Internal handles were changed. We need descriptor updates
        descUpdateQueued.trigger();
    }

    void growUntil(int factor, VkDeviceSize finalSize) {
        auto oldCapacity = capacity;

        capacity *= factor;
        capacity = (capacity + alignment - 1) & ~(alignment - 1);

        // Notice how we don't grow the staging buffer. This can cause problems when trying to download the buffer to host, but eeh
        auto newBuffer = allocator->createBuffer(capacity,
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        allocator->copyBuffer(buffer, newBuffer.first, oldCapacity, 0, 0, true);

        allocator->init->disp.destroyBuffer(buffer, nullptr);
        allocator->init->disp.freeMemory(memory, nullptr);

        buffer = newBuffer.first;
        memory = newBuffer.second;

        for (auto& buffer : buffers) {
            buffer.buffer = newBuffer.first;
        }
        getBufferAddress();

		if (capacity >= finalSize) {
			// Internal handles were changed. We need descriptor updates
			descUpdateQueued.trigger();
        }
        else {
			growUntil(factor, finalSize);
        }
    }

    // ez operation.
    std::vector<T> pop_back() {
        if (buffers.empty()) {
            throw std::out_of_range("No buffers to pop");
        }
        auto buf = buffers.back();
        auto dataSize = buf.size();
        auto alignedSize = buf.alignedSize(alignment);

        allocator->copyBuffer(buffer, stagingBuffer, dataSize, buf.offset, 0);

        auto data = std::vector<T>(buf.numElements);
        std::memcpy(data.data(), mapped, dataSize);
        std::memset(mapped, 0, dataSize);
        buffers.pop_back();
        offset -= alignedSize;
        occupied -= alignedSize;
        return data;
    }

    // expensive operation if instaClean is turned on. Use with caution.
    std::vector<T> erase(uint32_t index, bool instaClean = true) {
        if (index >= buffers.size()) {
            throw std::out_of_range("Index out of range");
        }
        if (buffers.size <= 0) {
            throw std::exception("MemPool has nothing left to erase.");
        }
        auto& buf = buffers[index];
        auto alignedSize = buf.alignedSize(alignment);
		auto elementSize = buf.numElements;
        auto dataSize = buf.size();
        // Copy Data from GPU Buffer to Staging Buffer
        allocator->copyBuffer(buffer, stagingBuffer, dataSize, buf.offset, 0, true);
        std::vector<T> output(buf.numElements);
        std::memcpy(output.data(), mapped, dataSize);
        std::memset(mapped, 0, alignedSize);

        // free buffer memory and manage offsets accordingly
        if (instaClean) {
            // free the blob of erased memory, push the remaining memory towards the left and zero out the tail of straggling memory
            allocator->freeMemory(buffer, memory, ((buf.offset + alignment - 1) & ~(alignment - 1)), alignedSize);
            // Update offset for next allocation
            offset -= alignedSize;
            occupied -= alignedSize;
            // Update the offset of the remaining buffers
            for (size_t i = index; i < buffers.size(); ++i) {
                buffers[i].offset -= alignedSize;
				elementOffset -= elementSize;
            }

            // internal offsets were changed. We need descriptor updates
            descUpdateQueued.trigger();
        }
        // just record the gap and leave it there. The record of the gaps can be cleaned up later. Kind of like a garbage collector.
        else {
            deadMemory += alignedSize;
        }
        // Remove the buffer from the vector
        buffers.erase(buffers.begin() + index);

        // update binding index of the remaining buffers. You need to recreate descriptor sets after this. erase is a trash operation, I know.
        for (size_t in = index; in < buffers.size(); ++in) {
            buffers[in].createDescriptors(buffers[in].bindingIndex - 1, flags);
        }

        descUpdateQueued.trigger();
        return output;
    }

    // garbage collector (kinda)
    void cleanGaps() {

        if (deadMemory <= 0) { throw std::exception("MemPool is already clean."); }

        // record the good stuff (aligned to vulkan's reqs)
        // offset, range format. both need to be aligned to vulkan's reqs. Automatically done by the MemPool
        std::vector<std::pair<VkDeviceSize, VkDeviceSize>> alive;
        for (auto& buffer : buffers) {
            auto alignedSize = buffer.alignedSize(alignment);
            alive.push_back({ buffer.offset, alignedSize });
        }
        // defragment the memory
        allocator->defragment(buffer, memory, alive);

        // update the offset of the remaining buffers
        VkDeviceSize runningOffset = 0;
        uint32_t runningElementOffset = 0;
        for (auto& buffer : buffers) {
            auto alignedSize = buffer.alignedSize(alignment);
            buffer.offset = runningOffset;
			buffer.elementOffset = runningElementOffset;
            runningOffset += alignedSize;
			runningElementOffset += buffer.numElements;
        }

        // need to notify all user resources to update their descriptor sets
        descUpdateQueued.trigger();

        // decrement the offset by the amount of memory we've cleared
        offset -= deadMemory;
        occupied -= deadMemory;
    }

    // expensive operation. Use with caution. replaces element[index] with given element
    void replace(uint32_t index, const std::vector<T>& data) {
        if (index > buffers.size()) {
            throw std::out_of_range("Index out of range");
        }

        const VkDeviceSize dataSize = data.size() * sizeof(T);
        const VkDeviceSize alignedSize = (dataSize + alignment - 1) & ~(alignment - 1);

        // Check if there's enough space
        if (((offset + alignedSize) - buffers[index].alignedSize(alignment)) > capacity) {
            grow(2);
        }

        auto alignedOffset = buffers[index].offset;
        auto deadBufferSize = buffers[index].alignedSize(alignment);
        auto deadBufferElementSize = buffers[index].numElements;

        // decrement the offset by the buffer we're replacing
        offset -= deadBufferSize;
        occupied -= deadBufferSize;

        // we're using the staging buffer to hold the toReplace memory
        std::memcpy(mapped, data.data(), dataSize);
        allocator->replaceMemory(buffer, memory, stagingBuffer, alignedSize, alignedOffset);
        std::memset(mapped, 0, data.size() * sizeof(T));

        // Edit the buffer element to reflect the new data it now represents
        buffers[index].numElements = static_cast<uint32_t>(data.size());

        // increment the offset by the buffer we've replaced
        offset += alignedSize;
        occupied += alignedSize;

        for (size_t i = index + 1; i < buffers.size(); ++i) {
            buffers[i].offset -= deadBufferSize; // decrement the offset of the remaining buffers by the dead buffer's size
			buffers[i].elementOffset -= deadBufferElementSize;
            buffers[i].offset += alignedSize;    // increment the offset of the remaining buffers by the new buffer's size
			buffers[i].elementOffset += static_cast<uint32_t>(data.size());
            buffers[i].createDescriptors(static_cast<uint32_t>(i), flags); // assign correct binding
        }

        // internal offsets were changed. We need descriptor updates
        descUpdateQueued.trigger();
    }

    void replace(uint32_t index, const StandaloneBuffer<T>& data) {
        if (index > buffers.size()) {
            throw std::out_of_range("Index out of range");
        }

        const VkDeviceSize alignedSize = data.capacity;

        // Check if there's enough space
        if (((offset + alignedSize) - buffers[index].alignedSize(alignment)) > capacity) {
            grow(2);
        }

        auto alignedOffset = buffers[index].offset;
        auto deadBufferSize = buffers[index].alignedSize(alignment);
		auto deadBufferElementSize = buffers[index].numElements;
        // decrement the offset by the buffer we're replacing
        offset -= deadBufferSize;
        occupied -= deadBufferSize;

        allocator->replaceMemory(buffer, memory, data.buffer, alignedSize, alignedOffset);

        // Edit the buffer element to reflect the new data it now represents
        buffers[index].numElements = static_cast<uint32_t>(data.size());

        // Update offset for next allocation
        offset += alignedSize;
        occupied += alignedSize;

        for (size_t i = index + 1; i < buffers.size(); ++i) {
            buffers[i].offset -= deadBufferSize; // decrement the offset of the remaining buffers by the dead buffer's size
			buffers[i].elementOffset -= deadBufferElementSize;
            buffers[i].offset += alignedSize;    // increment the offset of the remaining buffers by the new buffer's size
			buffers[i].elementOffset += static_cast<uint32_t>(data.size());
            buffers[i].createDescriptors(static_cast<uint32_t>(i), flags); // assign correct binding
        }

        // internal offsets were changed. We need descriptor updates
        descUpdateQueued.trigger();
    }

    void insert(uint32_t index, std::vector<T>& data) {

        if (index > buffers.size() || index < 0) {
            throw std::out_of_range("Index out of range");
        }

        auto alignedSize = (data.size() * sizeof(T) + alignment - 1) & ~(alignment - 1);

        if (capacity < occupied + alignedSize) {
            grow(2);
        }

        // we're using the staging buffer to hold the toInsert memory
        std::memcpy(mapped, data.data(), data.size() * sizeof(T));
        allocator->insertMemory(buffer, memory, stagingBuffer, buffers[index].offset, alignedSize);
        std::memset(mapped, 0, data.size() * sizeof(T));

        // Create a Buffer entry
        Buffer<T> newBuffer;
        newBuffer.buffer = buffer;
        newBuffer.offset = buffers[index].offset;
        newBuffer.numElements = static_cast<uint32_t>(data.size());
		newBuffer.elementOffset = buffers[index - 1].elementOffset + buffers[index - 1].numElements;
        newBuffer.createDescriptors(index, flags);
        buffers.insert(buffers.begin() + index, newBuffer);

        // Update offset for next allocation
        offset += alignedSize;
        occupied += alignedSize;
        for (size_t i = index + 1; i < buffers.size(); ++i) {
            buffers[i].offset += alignedSize;
			buffers[i].elementOffset += static_cast<uint32_t>(data.size());
            buffers[i].createDescriptors(static_cast<uint32_t>(i), flags); // assign correct binding
        }

        // internal offsets were changed. We need descriptor updates
        descUpdateQueued.trigger();
    }

    void insert(uint32_t index, StandaloneBuffer<T>& data) {

        if (index > buffers.size() || index < 0) {
            throw std::out_of_range("Index out of range");
        }

        auto alignedSize = data.capacity;

        if (capacity < occupied + alignedSize) {
            grow(2);
        }

        allocator->insertMemory(buffer, memory, data.buffer, buffers[index].offset, alignedSize);

        // Create a Buffer entry
        Buffer<T> newBuffer;
        newBuffer.buffer = buffer;
        newBuffer.offset = buffers[index].offset;
		newBuffer.elementOffset = buffers[index - 1].elementOffset + buffers[index - 1].numElements;
        newBuffer.numElements = static_cast<uint32_t>(data.capacity / sizeof(T));
        newBuffer.createDescriptors(index, flags);
        buffers.insert(buffers.begin() + index, newBuffer);

        // Update offset for next allocation
        offset += alignedSize;
        occupied += alignedSize;
        for (size_t i = index + 1; i < buffers.size(); ++i) {
            buffers[i].offset += alignedSize;
			buffers[i].elementOffset += static_cast<uint32_t>(data.size());
            buffers[i].createDescriptors(static_cast<uint32_t>(i), flags); // assign correct binding
        }
        // internal offsets were changed. We need descriptor updates
        descUpdateQueued.trigger();
    }

    std::vector<T> downloadPool() {
        allocator->copyBuffer(buffer, stagingBuffer, capacity, 0, 0, true);
        std::vector<T> data;
        data.resize(capacity / sizeof(T));
        std::memcpy(data.data(), mapped, capacity);
        std::memset(mapped, 0, capacity);
        return data;
    }

    // Daggers in my eyes, but it's not that important, so not gonna waste time on it
    std::vector<std::vector<T>> downloadBuffers() {
        std::vector<std::vector<T>> bufs{};
        for (auto& buf : buffers) {
            allocator->copyBuffer(buffer, stagingBuffer, buf.size(), buf.offset, 0, true);
            std::vector<T> data(buf.numElements);
            std::memcpy(data.data(), mapped, buf.size());
            std::memset(mapped, 0, buf.size());
            bufs.push_back(data);
        }

        return bufs;
    }

    void printPool() {
        auto allBufs = downloadBuffers();

        for (int i = 0; i < allBufs.size(); i++) {
            std::cout << "buffer number " << i << "\n\t";
            for (auto& ele : allBufs[i]) {
                std::cout << ele << " ";
            }
            std::cout << "\n";
        }
    }
};

// uses the default descriptors of vulkan.
template<typename T>
class gpuTask {
private:
    // Vulkan handles:
    VkQueue queue;
    VkPipelineLayout pipeline_layout;
    VkPipeline compute_pipeline;
    VkCommandPool command_pool;
    VkCommandBuffer command_buffer;

    // the input buffers are inside the buffers pool
    MemPool<T> buffers;

    // the output is a standalone buffer
    StandaloneBuffer<T> output;

    // Vulkan stuff:
    VkDescriptorSet descriptor_set; // only one descriptor set per task
    VkDescriptorPool descriptor_pool; // the pool will only contain one desc_set
    VkDescriptorSetLayout descriptor_set_layout; // the layout will have bindings for all the inputs and then the output

    uint32_t taskNumber = 0;
    uint32_t numInputs = 0;
    uint32_t numOutputs = 0;

    Init& init; // global vulkan handles

    std::vector<char> spv_code = std::vector<char>{};// we'll have shader bytecode in this

    //get the compute queue family inside the GPU
    int get_queues() {
        auto gq = init.device.get_queue(vkb::QueueType::compute);
        if (!gq.has_value()) {
            std::cout << "failed to get queue: " << gq.error().message() << "\n";
            return -1;
        }
        queue = gq.value();
        return 0;
    }

    // create the descriptors for the inputs and the output. Inputs are readonly, output is writeonly
    void create_descriptors() {
        uint32_t total_buffers = static_cast<uint32_t>(buffers.buffers.size() + 1);

        // --- Descriptor Pool ---
        VkDescriptorPoolSize pool_size = {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            total_buffers
        };

        VkDescriptorPoolCreateInfo pool_info = {};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.maxSets = 1;
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes = &pool_size;
        init.disp.createDescriptorPool(&pool_info, nullptr, &descriptor_pool);

        // --- Descriptor Set Layout ---
        std::vector<VkDescriptorSetLayoutBinding> bindings(total_buffers);
        for (uint32_t i = 0; i < buffers.buffers.size(); ++i) {
            // the input descriptors are managed by the MemPool, so they're already created
            bindings[i] = buffers.buffers[i].binding;
        }
        output.bindingIndex = buffers.buffers.size();
        output.createDescriptors();
        bindings.back() = output.binding;

        VkDescriptorSetLayoutCreateInfo dsl_info = {};
        dsl_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dsl_info.bindingCount = total_buffers;
        dsl_info.pBindings = bindings.data();
        init.disp.createDescriptorSetLayout(&dsl_info, nullptr, &descriptor_set_layout);

        // --- Descriptor Set Allocation ---
        VkDescriptorSetAllocateInfo ds_allocate_info = {};
        ds_allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ds_allocate_info.descriptorPool = descriptor_pool;
        ds_allocate_info.descriptorSetCount = 1;
        ds_allocate_info.pSetLayouts = &descriptor_set_layout;
        init.disp.allocateDescriptorSets(&ds_allocate_info, &descriptor_set);

        // --- Update Descriptor Set ---
        std::vector<VkWriteDescriptorSet> write_desc_sets(total_buffers);

        // Input Buffers
        for (uint32_t i = 0; i < buffers.buffers.size(); ++i) {
            buffers.buffers[i].updateDescriptorSet(descriptor_set);
            write_desc_sets[i] = buffers.buffers[i].wrt_desc_set;
        }
        // output buffer
        output.updateDescriptorSet(descriptor_set);
        write_desc_sets.back() = output.wrt_desc_set;

        init.disp.updateDescriptorSets(write_desc_sets.size(), write_desc_sets.data(), 0, nullptr);
    }

    // create the compute pipeline for the gpu execution of this task
    void create_pipeline() {
        VkShaderModuleCreateInfo create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        create_info.codeSize = spv_code.size();
        create_info.pCode = reinterpret_cast<const uint32_t*>(spv_code.data());

        VkShaderModule shader_module;
        init.disp.createShaderModule(&create_info, nullptr, &shader_module);

        VkPipelineShaderStageCreateInfo shader_stage_info = {};
        shader_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shader_stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        shader_stage_info.module = shader_module;
        shader_stage_info.pName = "main";

        VkPipelineLayoutCreateInfo pipeline_layout_info = {};
        pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_info.setLayoutCount = 1;
        pipeline_layout_info.pSetLayouts = &descriptor_set_layout;
        pipeline_layout_info.pushConstantRangeCount = 0;
        init.disp.createPipelineLayout(&pipeline_layout_info, nullptr, &pipeline_layout);

        VkComputePipelineCreateInfo pipeline_info = {};
        pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeline_info.stage = shader_stage_info;
        pipeline_info.layout = pipeline_layout;
        pipeline_info.basePipelineHandle = VK_NULL_HANDLE;
        init.disp.createComputePipelines(VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &compute_pipeline);

        init.disp.destroyShaderModule(shader_module, nullptr);
    }

    // The command pool where we'll submit our commands for buffer recording and compute shader dispatching:
    void create_command_pool() {
        VkCommandPoolCreateInfo pool_info = {};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.queueFamilyIndex = init.device.get_queue_index(vkb::QueueType::compute).value();
        init.disp.createCommandPool(&pool_info, nullptr, &command_pool);

        VkCommandBufferAllocateInfo allocate_info = {};
        allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocate_info.commandPool = command_pool;
        allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocate_info.commandBufferCount = 1;
        init.disp.allocateCommandBuffers(&allocate_info, &command_buffer);
    }

public:
    gpuTask(std::vector<char>& spvByteCodeCompShader, Init& init, uint32_t numInputs) : spv_code(spvByteCodeCompShader), init(init), buffers(MemPool<T>(&init, numInputs)) {};
    gpuTask(gpuTask& base, uint32_t numInputs) {
        init = base.init;
        queue = base.queue;
        pipeline_layout = base.pipeline_layout;
        command_pool = base.command_pool;
        command_buffer = base.command_buffer;
        compute_pipeline = base.compute_pipeline;
        descriptor_pool = base.descriptor_pool;
        descriptor_set = base.descriptor_set;
        descriptor_set_layout = base.descriptor_set_layout;
        buffers = MemPool<T>(&init, numInputs);
    }
    // load floating point inputs into GPU memory using memory staging:
    void load_inputs(std::vector<std::vector<T>>& input_data) {
        // Ensure input vectors are properly sized
        size_t num_inputs = input_data.size();
        for (int i = 0; i < num_inputs; i++) {
            if (!buffers.push_back(input_data[i])) {
                throw std::runtime_error("Couldn't push_back into input buffer! Out of memory!");
            }
        }
        numInputs = num_inputs;

        create_descriptors();
        create_pipeline();
    }

    void updateBuffers() {
        numInputs = buffers.size();
        create_descriptors();
        create_pipeline();
    }

    void initiateCompute() {
        get_queues();
        create_command_pool();
        buffers.commandPool = command_pool;
        buffers.computeQueue = queue;

        // --- Create Output Buffer ---
        output.init = &init;
        output.computeQueue = queue;
        output.commandPool = command_pool;
        output.bindingIndex = buffers.buffers.size();
        auto outputBuf = std::vector<T>(10);
        output.alloc(outputBuf, 10);
    }

    std::vector<T> compute() {

        // --- Begin command buffer recording ---
        VkCommandBufferBeginInfo begin_info = {};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        init.disp.beginCommandBuffer(command_buffer, &begin_info);

        // Bind compute pipeline and descriptor set
        init.disp.cmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline);
        init.disp.cmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);

        // Dispatch compute shader
        uint32_t workgroup_count = 3;
        init.disp.cmdDispatch(command_buffer, workgroup_count, 1, 1);

        // End command buffer recording
        init.disp.endCommandBuffer(command_buffer);

        // Create Fence for GPU Synchronization
        VkFenceCreateInfo fence_info = {};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence compute_fence;
        init.disp.createFence(&fence_info, nullptr, &compute_fence);

        // Submit command buffer with fence
        VkSubmitInfo submit_info = {};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffer;
        init.disp.queueSubmit(queue, 1, &submit_info, compute_fence);

        // Wait for GPU to finish computation
        init.disp.waitForFences(1, &compute_fence, VK_TRUE, UINT64_MAX);

        // Retrieve results from output buffer
        std::vector<T> result(10);
        output.downloadBuffer(result, 3);

        // Cleanup fence
        init.disp.destroyFence(compute_fence, nullptr);
        cleanup();

        return result;
    }

    ~gpuTask() {}

private:
    void cleanup() {
        init.disp.destroyCommandPool(command_pool, nullptr);
        init.disp.destroyPipeline(compute_pipeline, nullptr);
        init.disp.destroyPipelineLayout(pipeline_layout, nullptr);
        init.disp.destroyDescriptorPool(descriptor_pool, nullptr);
        init.disp.destroyDescriptorSetLayout(descriptor_set_layout, nullptr);
    }
};
template<typename T>
struct ComputeSequence;

// The main power of the library. This is an element inside of the ComputeSequence, and it represents a Compute Shader
template<typename T>
class SequentialgpuTask {
    friend struct ComputeSequence<T>;
private:
    // Vulkan handles:
    VkQueue queue;
    VkPipelineLayout pipeline_layout;
    VkPipeline compute_pipeline;
    VkCommandPool command_pool;

    // the input buffers are inside the MemPool of the ComputeSequence
    std::vector<uint32_t> inputIndices; // the indices of the buffers stored inside the MemPool

    // the output buffers 
    std::vector<uint32_t> outputIndices; // the indices of the buffers stored inside the MemPool

    // Vulkan stuff (the ComputeSequence contains all of the pool stuff):
    VkDescriptorSet descriptor_set = {}; // only one descriptor set per task
    VkDescriptorPool descriptor_pool; // pool is managed by the ComputeSequence
    VkDescriptorSetLayout descriptor_set_layout; // the layout will have bindings for all the inputs and then the outputs

    uint32_t taskNumber = 0;
    uint32_t numInputs = 0;
    uint32_t numOutputs = 0;

    Init& init; // global vulkan handles

    std::vector<char> spv_code = std::vector<char>{};// we'll have shader bytecode in this

    // create the descriptors for the inputs and the outputs. Inputs are readonly, output is writeonly
    void create_descriptors(MemPool<T>& memPool) {
        uint32_t total_buffers = static_cast<uint32_t>(inputIndices.size() + outputIndices.size());

        // --- Descriptor Set Layout ---
        std::vector<VkDescriptorSetLayoutBinding> bindings(total_buffers);
        for (uint32_t i = 0; i < inputIndices.size(); i++) {
            bindings[i] = memPool.buffers[inputIndices[i]].binding;
        }
        for (uint32_t i = 0; i < outputIndices.size(); i++) {
            // this took me embarrasingly long to figure out... :(
            bindings[i + inputIndices.size()] = memPool.buffers[outputIndices[i]].binding;
        }

        VkDescriptorSetLayoutCreateInfo dsl_info = {};
        dsl_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dsl_info.bindingCount = total_buffers;
        dsl_info.pBindings = bindings.data();
        init.disp.createDescriptorSetLayout(&dsl_info, nullptr, &descriptor_set_layout);
    }

    // called after descriptor sets are allocated by the ComputeSequence
    void update_descriptors(MemPool<T>& memPool) {
        uint32_t total_buffers = static_cast<uint32_t>(inputIndices.size() + outputIndices.size());
        // --- Update Descriptor Set ---
        std::vector<VkWriteDescriptorSet> write_desc_sets(total_buffers);

        // Input Buffers
        for (uint32_t i = 0; i < inputIndices.size(); i++) {
            memPool.buffers[inputIndices[i]].updateDescriptorSet(descriptor_set);
            write_desc_sets[i] = memPool.buffers[inputIndices[i]].wrt_desc_set;
        }
        // output buffers
        for (uint32_t i = 0; i < outputIndices.size(); i++) {
            memPool.buffers[outputIndices[i]].updateDescriptorSet(descriptor_set);
            // this took me embarrasingly long to figure out... :(
            write_desc_sets[i + inputIndices.size()] = memPool.buffers[outputIndices[i]].wrt_desc_set;
        }

        init.disp.updateDescriptorSets(write_desc_sets.size(), write_desc_sets.data(), 0, nullptr);
    }

    // create the compute pipeline for the gpu execution of this task
    void create_pipeline() {
        VkShaderModuleCreateInfo create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        create_info.codeSize = spv_code.size();
        create_info.pCode = reinterpret_cast<const uint32_t*>(spv_code.data());

        VkShaderModule shader_module;
        init.disp.createShaderModule(&create_info, nullptr, &shader_module);

        VkPipelineShaderStageCreateInfo shader_stage_info = {};
        shader_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shader_stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        shader_stage_info.module = shader_module;
        shader_stage_info.pName = "main";

        VkPipelineLayoutCreateInfo pipeline_layout_info = {};
        pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_info.setLayoutCount = 1;
        pipeline_layout_info.pSetLayouts = &descriptor_set_layout;
        pipeline_layout_info.pushConstantRangeCount = 0;
        init.disp.createPipelineLayout(&pipeline_layout_info, nullptr, &pipeline_layout);

        VkComputePipelineCreateInfo pipeline_info = {};
        pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeline_info.stage = shader_stage_info;
        pipeline_info.layout = pipeline_layout;
        pipeline_info.basePipelineHandle = VK_NULL_HANDLE;
        init.disp.createComputePipelines(VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &compute_pipeline);

        init.disp.destroyShaderModule(shader_module, nullptr);
    }

public:
    SequentialgpuTask(std::vector<char>& spvByteCodeCompShader, Init& init) : spv_code(spvByteCodeCompShader), init(init) {};

    // load buffer indices. The buffers themselves are stored in one massive MemPool in the ComputeSequence for efficiency
    void load_indices(std::vector<uint32_t>& inputIndicess, std::vector<uint32_t>& outputIndicess) {
        numInputs = inputIndicess.size();
        inputIndices = inputIndicess;
        numOutputs = outputIndicess.size();
        outputIndices = outputIndicess;
    }

    // dispatch commands
    void compute(VkCommandBuffer& command_buffer, uint32_t workgroup_x, uint32_t workgroup_y, uint32_t workgroup_z) {
        // Bind compute pipeline and descriptor set
        init.disp.cmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline);
        init.disp.cmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);

        // Dispatch compute shader
        init.disp.cmdDispatch(command_buffer, workgroup_x, workgroup_y, workgroup_z);
    }

    ~SequentialgpuTask() {}

private:
    void cleanup() {
        init.disp.destroyCommandPool(command_pool, nullptr);
        init.disp.destroyPipeline(compute_pipeline, nullptr);
        init.disp.destroyPipelineLayout(pipeline_layout, nullptr);
        init.disp.destroyDescriptorPool(descriptor_pool, nullptr);
        init.disp.destroyDescriptorSetLayout(descriptor_set_layout, nullptr);
    }
};

// copy-pasted from the compute shader example of VkBootstrap
int device_initialization(Init& init) {
    vkb::InstanceBuilder instance_builder;
    auto instance_ret = instance_builder.use_default_debug_messenger()
        .request_validation_layers()
        .set_headless() // Skip vk-bootstrap trying to create WSI for you
        .build();
    if (!instance_ret) {
        std::cout << instance_ret.error().message() << "\n";
        return -1;
    }
    init.instance = instance_ret.value();

    init.inst_disp = init.instance.make_table();

    vkb::PhysicalDeviceSelector phys_device_selector(init.instance);
    auto phys_device_ret = phys_device_selector.select();
    if (!phys_device_ret) {
        std::cout << phys_device_ret.error().message() << "\n";
        return -1;
    }
    vkb::PhysicalDevice physical_device = phys_device_ret.value();

    vkb::DeviceBuilder device_builder{ physical_device };
    auto device_ret = device_builder.build();
    if (!device_ret) {
        std::cout << device_ret.error().message() << "\n";
        return -1;
    }
    init.device = device_ret.value();

    init.disp = init.device.make_table();

    return 0;
}

// Sequentially executes SequentialgpuTasks efficiently. 
// Uses one huge memory pool for all the inputs and outputs of the tasks so one task's output can be the next task's input
template<typename T>
struct ComputeSequence {
    std::vector<SequentialgpuTask<T>> tasks;
    Init* init;
    VkQueue queue;
    VkCommandPool commandPool;
    VkDescriptorPool descriptorPool;
    uint32_t totalNumBuffers;

    VkCommandBuffer command_buffer;
    MemPool<T> allBuffers;

    ComputeSequence(Init* init, uint32_t numBuffers) : init(init), allBuffers(init, numBuffers) {};

    //get the compute queue family inside the GPU
    int get_queues() {
        auto gq = init->device.get_queue(vkb::QueueType::compute);
        if (!gq.has_value()) {
            std::cout << "failed to get queue: " << gq.error().message() << "\n";
            return -1;
        }
        queue = gq.value();
        return 0;
    }

    void create_command_pool() {
        VkCommandPoolCreateInfo pool_info = {};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.queueFamilyIndex = init->device.get_queue_index(vkb::QueueType::compute).value();
        init->disp.createCommandPool(&pool_info, nullptr, &commandPool);
    }

    void create_descriptor_pool() {
        // --- Descriptor Pool ---
        VkDescriptorPoolSize pool_size = {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            totalNumBuffers
        };

        VkDescriptorPoolCreateInfo pool_info = {};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.maxSets = tasks.size();
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes = &pool_size;
        init->disp.createDescriptorPool(&pool_info, nullptr, &descriptorPool);
    }

    void initializeCompute() {
        // create all the descriptors

        create_descriptor_pool();

        auto desc_set_layouts = std::vector<VkDescriptorSetLayout>(tasks.size());
        auto desc_sets = std::vector<VkDescriptorSet>(tasks.size());
        int i = 0;
        for (auto& task : tasks) {
            task.queue = queue;
            task.command_pool = commandPool;
            task.descriptor_pool = descriptorPool;
            task.create_descriptors(allBuffers);
            desc_set_layouts[i] = (task.descriptor_set_layout);
            desc_sets[i] = (task.descriptor_set);
            i++;
        }

        // allocate the descriptors
        VkDescriptorSetAllocateInfo ds_allocate_info = {};
        ds_allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ds_allocate_info.descriptorPool = descriptorPool;
        ds_allocate_info.descriptorSetCount = tasks.size();
        ds_allocate_info.pSetLayouts = desc_set_layouts.data();
        init->disp.allocateDescriptorSets(&ds_allocate_info, desc_sets.data());

        // update the descriptors
        i = 0;
        for (auto& task : tasks) {
            task.descriptor_set = desc_sets[i];
            task.update_descriptors(allBuffers);
            task.create_pipeline();
            i++;
        }
    }

    void initMemory(uint32_t totalNumInputss) {
        get_queues();
        create_command_pool();
        //allBuffers = MemPool<T>(init, totalNumInputss);
        allBuffers.computeQueue = queue;
        allBuffers.commandPool = commandPool;
    }

    void addBuffers(std::vector<std::vector<T>>& inputs) {
        for (int i = 0; i < inputs.size(); i++) {
            if (!allBuffers.push_back(inputs[i])) {
                throw std::runtime_error("couldn't add buffers to compute sequence");
            }
        }
    }

    void addTask(SequentialgpuTask<T>& task) {
        tasks.push_back(task);
    }

    // do the tasks sequentially for given iterations
    void doTasks(uint32_t iterations, uint32_t workgroup_x, uint32_t workgroup_y, uint32_t workgroup_z) {
        std::vector<VkCommandBuffer> command_buffers(tasks.size());
        std::vector<VkSemaphore> semaphores(tasks.size());

        // Allocate command buffers
        VkCommandBufferAllocateInfo alloc_info = {};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.commandPool = commandPool;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandBufferCount = static_cast<uint32_t>(tasks.size());
        init->disp.allocateCommandBuffers(&alloc_info, command_buffers.data());

        // Create semaphores
        VkSemaphoreCreateInfo semaphore_info = {};
        semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        for (size_t i = 0; i < tasks.size(); i++) {
            init->disp.createSemaphore(&semaphore_info, nullptr, &semaphores[i]);
        }

        // Record commands
        for (size_t i = 0; i < tasks.size(); i++) {
            VkCommandBufferBeginInfo begin_info = {};
            begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

            init->disp.beginCommandBuffer(command_buffers[i], &begin_info);
            tasks[i].compute(command_buffers[i], workgroup_x, workgroup_y, workgroup_z);
            init->disp.endCommandBuffer(command_buffers[i]);
        }

        // Submit tasks with semaphore chaining
        for (uint32_t iter = 0; iter < iterations; iter++) {
            for (size_t i = 0; i < tasks.size(); i++) {
                VkSubmitInfo submit_info = {};
                submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                submit_info.commandBufferCount = 1;
                submit_info.pCommandBuffers = &command_buffers[i];

                VkPipelineStageFlags wait_stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

                // First task has no semaphore dependency
                if (i == 0) {
                    submit_info.waitSemaphoreCount = 0;
                    submit_info.pWaitSemaphores = nullptr;
                }
                else {
                    submit_info.waitSemaphoreCount = 1;
                    submit_info.pWaitSemaphores = &semaphores[i - 1];
                    submit_info.pWaitDstStageMask = &wait_stages;
                }

                submit_info.signalSemaphoreCount = 1;
                submit_info.pSignalSemaphores = &semaphores[i];

                init->disp.queueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE);
            }
        }

        // Wait for the last task to finish
        init->disp.queueWaitIdle(queue);

        // Cleanup
        for (size_t i = 0; i < tasks.size(); i++) {
            init->disp.destroySemaphore(semaphores[i], nullptr);
        }
        init->disp.freeCommandBuffers(commandPool, static_cast<uint32_t>(tasks.size()), command_buffers.data());
    }
};

// Read shader bytecode from a file
std::vector<char> readShaderBytecode(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);  // Open file at the end in binary mode

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open shader file: " + filename);
    }

    size_t fileSize = file.tellg();  // Get file size
    std::vector<char> buffer(fileSize);

    file.seekg(0);  // Go back to the beginning
    file.read(buffer.data(), fileSize);  // Read file into buffer
    file.close();

    return buffer;
}
