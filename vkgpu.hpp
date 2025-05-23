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

    ~Allocator() {
        if (!commandBuffers.empty()) {
            init->disp.freeCommandBuffers(commandPool, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());
			commandBuffers.clear();
        }
		
		init->disp.destroyCommandPool(commandPool, nullptr);
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

        if (init->disp.allocateMemory(&allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
            throw std::runtime_error("could not allocate memory");
        }
        if (init->disp.bindBufferMemory(buffer, bufferMemory, 0) != VK_SUCCESS) {
            throw std::runtime_error("could not bind memory");
        }
        return { buffer, bufferMemory };
    };

	std::pair<VkImage, VkDeviceMemory> createImage(VkDeviceSize size, VkImageUsageFlags usage, VkImageType imageType, VkMemoryPropertyFlags properties, VkFormat format = VK_FORMAT_R8G8B8A8_UNORM,VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT, VkSharingMode sharingMode = VK_SHARING_MODE_EXCLUSIVE) {
		VkImage image{};
		VkDeviceMemory imageMemory{};
		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = imageType;
		imageInfo.extent.width = static_cast<uint32_t>(size);
		imageInfo.extent.height = static_cast<uint32_t>(size);
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.format = format;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
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
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingMemory;
        VkDeviceSize bufferSize;

        // Get memory requirements for the buffer
        VkMemoryRequirements memRequirements;
        init->disp.getBufferMemoryRequirements(buffer, &memRequirements);
        bufferSize = memRequirements.size;

        // Create a staging buffer
        auto stageBuff = createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        stagingBuffer = stageBuff.first;
        stagingMemory = stageBuff.second;

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

    void createMemoryBlob(VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize size) {
		// Step 1: Create a staging buffer to temporarily hold the data
		VkBuffer stagingBuffer;
		VkDeviceMemory stagingMemory;
		VkDeviceSize bufferSize;
		// Get memory requirements for the buffer
		VkMemoryRequirements memRequirements;
		init->disp.getBufferMemoryRequirements(buffer, &memRequirements);
		bufferSize = memRequirements.size;
		// Create a staging buffer
		auto stageBuff = createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		stagingBuffer = stageBuff.first;
		stagingMemory = stageBuff.second;
		// Step 2: Copy data before the gap to the staging buffer
		if (offset > 0) {
            sequentialCopyBuffer(buffer, stagingBuffer, offset, 0, 0);
		}
		// Step 3: Copy data after the gap to the staging buffer
		VkDeviceSize remainingSize = bufferSize - (offset + size);
		if (remainingSize > 0) {
            sequentialCopyBuffer(buffer, stagingBuffer, remainingSize, offset + size, offset);
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

    // this is cancer
    void defragment(VkBuffer buffer, VkDeviceMemory memory) {
		// not gonna implement this. This is a nightmare.
    }

    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size, VkDeviceSize srcOffset, VkDeviceSize dstOffset, bool async = false) {
        auto cmd = beginSingleTimeCommands();
        VkBufferCopy copyRegion{};
        copyRegion.size = size;
        copyRegion.dstOffset = dstOffset;
        copyRegion.srcOffset = srcOffset;
        vkCmdCopyBuffer(commandBuffers.back(), srcBuffer, dstBuffer, 1, &copyRegion);
        endSingleTimeCommands(async);
    }

	void sequentialCopyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size, VkDeviceSize srcOffset, VkDeviceSize dstOffset) {
        auto cmd = beginSingleTimeCommands();
		VkBufferCopy copyRegion{};
		copyRegion.size = size;
		copyRegion.dstOffset = dstOffset;
		copyRegion.srcOffset = srcOffset;
		vkCmdCopyBuffer(commandBuffers.back(), srcBuffer, dstBuffer, 1, &copyRegion);
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

private:
    Init* init;
    VkQueue allocQueue;
    VkCommandPool commandPool;
    std::vector<VkCommandBuffer> commandBuffers;
    
    int beginSingleTimeCommands() {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = commandPool;
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

    void endSingleTimeCommands(bool async = false, bool dispatch = true) {
        if (commandBuffers.empty()) return;

        auto cmd = commandBuffers.back();
        init->disp.endCommandBuffer(cmd);

        if (dispatch) {
            VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &cmd;
            init->disp.queueSubmit(allocQueue, 1, &submitInfo, VK_NULL_HANDLE);
            if (!async) {
                init->disp.deviceWaitIdle();
            }
            else {
                init->disp.queueWaitIdle(allocQueue);
            }
            init->disp.freeCommandBuffers(commandPool, 1, &cmd);
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

    VkBuffer stagingBuffer{};
	VkDeviceMemory stagingBufferMemory{}; // Staging memory on CPU (same size as buffer and memory on GPU)

    VkDeviceSize alignment;
    VkDeviceSize capacity;

    // stuff you need to send to pipeline creation:
    VkDescriptorSetLayoutBinding binding{};
    VkWriteDescriptorSet wrt_desc_set{};
    VkDescriptorBufferInfo desc_buf_info{};
    uint32_t bindingIndex;
    Init* init;

	// for copying buffers to GPU
	std::shared_ptr<Allocator> allocator;

    void* memMap;

    StandaloneBuffer(Init* init, uint32_t numElements, std::shared_ptr<Allocator>& allocator) : init(init), allocator(allocator) {

        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(init->device.physical_device, &props);
        alignment = props.limits.minStorageBufferOffsetAlignment;

        capacity = numElements * sizeof(T);
        capacity = (capacity + alignment - 1) & ~(alignment - 1);

        // Create the staging buffer
		auto stageBuff = allocator->createBuffer(capacity, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, true);
		stagingBuffer = stageBuff.first;
		stagingBufferMemory = stageBuff.second;
		
        // Create the buffer
        auto buff = allocator->createBuffer(capacity, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, true);
		buffer = buff.first;
		bufferMemory = buff.second;
        init->disp.mapMemory(stagingBufferMemory, 0, capacity, 0, &memMap);
    }
    StandaloneBuffer() : init(nullptr) { std::cout << "Called def constructor\n"; }

    void createDescriptors() {
        binding.binding = bindingIndex;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
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

    void clearBuffer() {
		init->disp.unmapMemory(stagingBufferMemory);
        init->disp.destroyBuffer(buffer, nullptr);
        init->disp.freeMemory(bufferMemory, nullptr);
		init->disp.destroyBuffer(stagingBuffer, nullptr);
		init->disp.freeMemory(stagingBufferMemory, nullptr);
    }

    void alloc(std::vector<T>& data, uint32_t numElements) {
        auto sizeOfData = static_cast<uint32_t>(sizeof(T) * numElements);
        std::memcpy(memMap, data.data(), sizeOfData);
        allocator->copyBuffer(stagingBuffer, buffer, sizeOfData, 0, 0, true);
		std::memset(memMap, 0, sizeOfData);
    }

    void downloadBuffer(std::vector<T>& data, uint32_t numElements) {
        VkDeviceSize bufferSize = sizeof(T) * numElements;
        allocator->copyBuffer(buffer, stagingBuffer, bufferSize, 0, 0, true);
        data.resize(numElements);
        std::memcpy(data.data(), memMap, bufferSize);
		std::memset(memMap, 0, bufferSize);
    }

    ~StandaloneBuffer() {
        clearBuffer();
    }
};

// an element inside the MemPool.
template<typename T>
struct Buffer {
    VkBuffer buffer;                      // Points to the MemPool's buffer
    //VkDeviceAddress bufferDeviceAddress;  // provided by MemPool
    VkDeviceSize offset;                  // Offset within the buffer
    uint32_t numElements;                 // Number of elements in this buffer

    // Descriptor set members (used when MemPool is not using bypassDescriptors)
    uint32_t bindingIndex;
    VkDescriptorSetLayoutBinding binding{};
    VkWriteDescriptorSet wrt_desc_set{};
    VkDescriptorBufferInfo desc_buf_info{};

    void createDescriptors(uint32_t bindingIdx) {
        bindingIndex = bindingIdx;
        binding.binding = bindingIndex;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
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

// like std::vector, but worse. And for the GPU
template<typename T>
struct MemPool {
    VkBuffer buffer;               // Single buffer for all allocations on GPU
    VkDeviceMemory memory;         // Backing memory on GPU
    
    // persistent staging buffer for efficiency. We should avoid allocating new memory whenever we can.
    // Not really the best solution, cause for every MemPool, 
    // we now have two memories of the same size that's taking up space in the computer... but eeh
    VkBuffer stagingBuffer;                  // Staging buffer on CPU (same size as buffer and memory on GPU)
    VkDeviceMemory stagingMemory;            // Staging memory on CPU (same size as buffer and memory on GPU)
    
    VkDeviceAddress bufferDeviceAddress; // Actual pointer to the memory inside of the GPU. used when bypassDescriptors are activated

    void* mapped = nullptr;        // Mapped pointer. But "memory" is device local. This is used to read from index
    Init* init;                    // Vulkan initialization context
    VkDeviceSize alignment;        // Buffer alignment requirement
    VkDeviceSize capacity;         // Total capacity in bytes
    VkDeviceSize offset = 0;       // Current allocation offset

    std::vector<Buffer<T>> buffers; // Track all allocated buffers
    bool bypassDescriptors;

    std::shared_ptr<Allocator> allocator;

    // maxElements means number of individual elements of type T inside of all the buffers. 
    // Meaning, if maxElements is set to 10, pool can sustain 10 buffers with 1 float each (if T = float).
    // But only 2 buffers with 5 floats each.
    MemPool(Init* init, uint32_t maxElements, bool bypassDescriptors = false, std::shared_ptr<Allocator>& allocator = nullptr) : init(init), bypassDescriptors(bypassDescriptors), allocator(allocator) {
        // Query alignment requirement for storage buffers
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(init->device.physical_device, &props);
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
		init->disp.mapMemory(stagingMemory, 0, capacity, 0, &mapped);

        auto buff = allocator->createBuffer(capacity,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		buffer = buff.first;
		memory = buff.second;
    }

    MemPool() {};

    ~MemPool() {
        init->disp.unmapMemory(stagingMemory);
        init->disp.destroyBuffer(buffer, nullptr);
        init->disp.freeMemory(memory, nullptr);
        init->disp.destroyBuffer(stagingBuffer, nullptr);
        init->disp.freeMemory(stagingMemory, nullptr);
    }

	size_t size() {
		return buffers.size();
	}

    bool push_back(const std::vector<T>& data, bool autoBind = true) {
        auto bindingIndex = buffers.size();
        const VkDeviceSize dataSize = data.size() * sizeof(T);
        const VkDeviceSize alignedSize = (dataSize + alignment - 1) & ~(alignment - 1);

        // Check if there's enough space
        if (offset + alignedSize > capacity) {
			grow(2);
        }

        // Step 1: Copy Data to Staging Buffer
        std::memcpy(mapped, data.data(), dataSize);
        
        // Step 2: Copy Staging Buffer to GPU Buffer
        allocator->copyBuffer(stagingBuffer, buffer, alignedSize, 0, offset, true);

        // Clear the staging buffer memory
        std::memset(mapped, 0, alignedSize);

        // Create a Buffer entry
        Buffer<T> newBuffer;
        newBuffer.buffer = buffer;
        newBuffer.offset = offset;
        newBuffer.numElements = static_cast<uint32_t>(data.size());
        if (autoBind && !bypassDescriptors) {
            newBuffer.createDescriptors(bindingIndex);
        }
        else if (bypassDescriptors && !autoBind){
            //newBuffer.bufferDeviceAddress = bufferDeviceAddress;
        }
        buffers.push_back(newBuffer);

        // Update offset for next allocation
        offset += alignedSize;
        return true;
    }

    // Retrieve data from GPU
    std::vector<T> operator[](size_t index) {
        if (index >= buffers.size()) {
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

	void resize(int newSize) {
        // Calculate total size with alignment
        capacity = newSize * sizeof(T);
        capacity = (capacity + alignment - 1) & ~(alignment - 1);

		auto newBuffer = allocator->createBuffer(capacity,
			VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		allocator->copyBuffer(buffer, newBuffer.first, capacity, 0, 0, true);

		init->disp.destroyBuffer(buffer, nullptr);
		init->disp.freeMemory(memory, nullptr);

		buffer = newBuffer.first;
		memory = newBuffer.second;

        for (auto& buffer : buffers) {
            buffer.buffer = newBuffer.first;
        }
	}

    void grow(int factor) {
        // Calculate total size with alignment
        auto oldCapacity = capacity;

        capacity = (buffers.size() + 10) * factor * sizeof(T);
        capacity = (capacity + alignment - 1) & ~(alignment - 1);

        auto newBuffer = allocator->createBuffer(capacity,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        allocator->copyBuffer(buffer, newBuffer.first, oldCapacity, 0, 0, true);

        init->disp.destroyBuffer(buffer, nullptr);
        init->disp.freeMemory(memory, nullptr);

        buffer = newBuffer.first;
        memory = newBuffer.second;

        for (auto& buffer : buffers) {
			buffer.buffer = newBuffer.first;
        }
    }

	std::vector<T> pop_back() {
		if (buffers.empty()) {
			throw std::out_of_range("No buffers to pop");
		}
		auto buf = buffers.back();
		auto dataSize = buf.numElements * sizeof(T);
		auto alignedSize = (dataSize + alignment - 1) & ~(alignment - 1);

		allocator->copyBuffer(buffer, stagingBuffer, dataSize, buf.offset, 0);

		auto data = std::vector<T>(buf.numElements);
		std::memcpy(data.data(), mapped, dataSize);
		std::memset(mapped, 0, alignedSize);
		buffers.pop_back();
		offset -= alignedSize;

		return data;
	}

    std::vector<T> erase(uint32_t index) {
        if (index >= buffers.size()) {
            throw std::out_of_range("Index out of range");
        }
        auto& buf = buffers[index];
        auto dataSize = buf.numElements * sizeof(T);
        auto alignedSize = (dataSize + alignment - 1) & ~(alignment - 1);
        // Copy Data from GPU Buffer to Staging Buffer
        allocator->copyBuffer(buffer, stagingBuffer, dataSize, buf.offset, 0, true);
        std::vector<T> output(buf.numElements);
        std::memcpy(output.data(), mapped, dataSize);
        std::memset(mapped, 0, alignedSize);

        // free the blob of empty memory, push the remaining memory towards the left and zero out the tail of straggling memory
		allocator->freeMemory(buffer, memory, ((buf.offset + alignment - 1) & ~(alignment - 1)), alignedSize);
        
        // Remove the buffer from the vector
		buffers.erase(buffers.begin() + index);
		
        // Update offset for next allocation
		offset -= alignedSize;
        // update binding index of the remaining buffers. You need to recreate descriptor sets after this. erase is a trash operation, I know.
        for (size_t in = index; in < buffers.size(); ++in) {
			buffers[in].createDescriptors(buffers[in].bindingIndex - 1);
		}
		// Update the offset of the remaining buffers
		for (size_t i = index; i < buffers.size(); ++i) {
			buffers[i].offset -= alignedSize;
		}

		return output;
    }

    void insert(uint32_t index, const std::vector<T>& data) {
        if (index > buffers.size()) {
            throw std::out_of_range("Index out of range");
        }
        auto bindingIndex = buffers.size();
        const VkDeviceSize dataSize = data.size() * sizeof(T);
        const VkDeviceSize alignedSize = (dataSize + alignment - 1) & ~(alignment - 1);
        // Check if there's enough space
        if (offset + alignedSize > capacity) {
            return; // Out of memory
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
