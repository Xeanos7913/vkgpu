// DO NOT USE! EXETREMELY BUGGY AND STILL IN DEVELOPMENT! will provide docs later

#pragma once

#define STB_IMAGE_IMPLEMENTATION
#include "VkBootstrap.h"
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_core.h>
#include <typeinfo>
#include <iostream>
#include <algorithm>
#include <vector>
#include <chrono>
#include <fstream>
#include <unordered_map>
#include "signal.hpp"
#include "stb_image.h"

#define GLM_FORCE_RADIANS  
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#define GLFW_INCLUDE_VULKAN
#define TINYOBJLOADER_IMPLEMENTATION

constexpr int MAX_FRAMES_IN_FLIGHT = 3;

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>  
#include <glm/gtc/matrix_transform.hpp>  
#include <glm/gtc/type_ptr.hpp>  
#include <glm/gtx/hash.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>
#include "tiny_obj_loader.h"

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

        for (auto& [buf, mem] : allocated) {
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

    std::pair<VkImage, VkDeviceMemory> createImage(VkDeviceSize width, VkDeviceSize height, uint32_t mipLevels, VkImageUsageFlags usage, VkImageType imageType, VkMemoryPropertyFlags properties, VkFormat format = VK_FORMAT_R8G8B8A8_UNORM, VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT, VkSharingMode sharingMode = VK_SHARING_MODE_EXCLUSIVE) {
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
        bufferAddress(other.bufferAddress) {
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

    void createDescriptors(uint32_t bindingIdx, VkShaderStageFlags flags = VK_SHADER_STAGE_COMPUTE_BIT) {
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

// "V" is your vertex struct with all the attribs
template<typename V>
struct VertexBuffer {
    Allocator* allocator;
    VkBuffer vBuffer;
    VkDeviceMemory vMem;
    VkDeviceAddress address;

    VkBuffer stageBuf;
    VkDeviceMemory stageMem;

    void* map;

    VertexBuffer() {};

    bool uploaded = false;

    VertexBuffer(Allocator* allocator, size_t numVertex) : allocator(allocator) {
        auto [buf, mem] = allocator->createBuffer(numVertex * sizeof(V), VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        vBuffer = buf;
        vMem = mem;

        auto [sBuf, sMem] = allocator->createBuffer(numVertex * sizeof(V), VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        stageBuf = sBuf;
        stageMem = sMem;

        allocator->init->disp.mapMemory(stageMem, 0, numVertex * sizeof(V), 0, &map);
        uploaded = true;

        VkBufferDeviceAddressInfoEXT bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_EXT;
        bufferInfo.buffer = vBuffer;
        bufferInfo.pNext = nullptr;
        address = allocator->init->disp.getBufferDeviceAddress(&bufferInfo);
    };

    void upload(std::vector<V>& vertices) {
        std::memcpy(map, vertices.data(), vertices.size() * sizeof(V));
        allocator->copyBuffer(stageBuf, vBuffer, vertices.size() * sizeof(V), 0, 0, true);
        std::memset(map, 0, vertices.size() * sizeof(V));
        uploaded = true;
    }

    ~VertexBuffer() {
        allocator->init->disp.unmapMemory(stageMem);
    }
};

template<typename T>
struct UniformBuffer {
    VkBuffer buffer;
    VkDeviceMemory memory;

    VkDeviceAddress address;
    VkBuffer stageBuffer;
    VkDeviceMemory stageMemory;

    VkDeviceSize capacity;

    void* map;
    Allocator* allocator;
    UniformBuffer() {};

    uint32_t bindingIndex;
    VkDescriptorSetLayoutBinding binding{};
    VkWriteDescriptorSet wrt_desc_set{};
    VkDescriptorBufferInfo desc_buf_info{};

    UniformBuffer(Allocator* allocator, size_t size) : allocator(allocator) {
        auto [buf, mem] = allocator->createBuffer(size * sizeof(T), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        buffer = buf;
        memory = mem;
        auto [sBuf, sMem] = allocator->createBuffer(size * sizeof(T), VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        stageBuffer = sBuf;
        stageMemory = sMem;
        capacity = size * sizeof(T);
        allocator->init->disp.mapMemory(stageMemory, 0, size * sizeof(T), 0, &map);
    };
    void upload(T& data) {
        std::memcpy(map, &data, sizeof(T));
        allocator->copyBuffer(stageBuffer, buffer, sizeof(T), 0, 0, true);
        std::memset(map, 0, sizeof(T));
    }

    void createDescriptors(uint32_t bindingIdx, VkShaderStageFlags flags = VK_SHADER_STAGE_ALL) {
        bindingIndex = bindingIdx;
        binding.binding = bindingIndex;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        binding.descriptorCount = 1;
        binding.stageFlags = flags;
        binding.pImmutableSamplers = nullptr;
    }

    void updateDescriptorSet(VkDescriptorSet& set) {
        desc_buf_info.buffer = buffer;
        desc_buf_info.offset = 0;
        desc_buf_info.range = capacity;

        wrt_desc_set.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wrt_desc_set.dstSet = set;
        wrt_desc_set.dstBinding = bindingIndex;
        wrt_desc_set.dstArrayElement = 0;
        wrt_desc_set.descriptorCount = 1;
        wrt_desc_set.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        wrt_desc_set.pBufferInfo = &desc_buf_info;
    }

    ~UniformBuffer() {
        allocator->init->disp.unmapMemory(stageMemory);
    }
};

typedef struct ImageDesc {
    uint32_t bindingIndex, width, height, mipLevels;
    VkFormat format;
    VkImageUsageFlags usage;
    VkImageLayout layout;
    VkSampleCountFlagBits samples;
    VkShaderStageFlags stage;
} ImageDesc;

struct Image2D {
    VkImage image;
    VkImageLayout imageLayout;
    VkDeviceMemory memory;
    VkImageView imageView;
    VkSampler sampler;

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingMemory;

    VkFormat format;
    VkImageUsageFlags usage;
    VkSampleCountFlagBits samples;
    uint32_t mipLevels;

    Allocator* alloc;

    VkDescriptorImageInfo imageInfo;
    uint32_t bindingIndex;
    VkDescriptorSetLayoutBinding binding{};
    VkWriteDescriptorSet wrt_desc_set{};

    int width, height;

    Image2D(int width, int height, int mipLevels, VkFormat format, VkImageUsageFlags usage, Allocator* allocator, VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT) : format(format), usage(usage), samples(samples), mipLevels(mipLevels), width(width), height(height), alloc(allocator) {
        imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; // need to change this to be customizable
        image = VK_NULL_HANDLE;
        memory = VK_NULL_HANDLE;
        imageView = VK_NULL_HANDLE;
        sampler = VK_NULL_HANDLE;
        auto image = allocator->createImage(width, height, mipLevels, usage | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_TYPE_2D, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, format, samples);
        this->image = image.first;
        this->memory = image.second;

        auto [buf, mem] = allocator->createBuffer(static_cast<VkDeviceSize>(width * height) * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        stagingBuffer = buf;
        stagingMemory = mem;

        if (format == VK_FORMAT_D32_SFLOAT) {
            //imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            createImageView(true);
        }
        else if (format == VK_FORMAT_D24_UNORM_S8_UINT) {
            //imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            createImageView(true);
        }

        else { createImageView(); }
        createSampler();
    };

    Image2D(ImageDesc desc, Allocator* allocator) : bindingIndex(desc.bindingIndex), format(desc.format), usage(desc.usage), samples(desc.samples), mipLevels(desc.mipLevels), width(desc.width), height(desc.height), alloc(allocator) {
        imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;  // need to change this to be customizable
        image = VK_NULL_HANDLE;
        memory = VK_NULL_HANDLE;
        imageView = VK_NULL_HANDLE;
        sampler = VK_NULL_HANDLE;
        auto image = allocator->createImage(width, height, mipLevels, usage, VK_IMAGE_TYPE_2D, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, format, samples);
        this->image = image.first;
        this->memory = image.second;

        auto [buf, mem] = allocator->createBuffer(static_cast<VkDeviceSize>(width * height) * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        stagingBuffer = buf;
        stagingMemory = mem;

        createImageView();
        createSampler();
    };

    Image2D(const std::string& path, Allocator* allocator) : alloc(allocator) {
        imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        image = VK_NULL_HANDLE;
        memory = VK_NULL_HANDLE;
        imageView = VK_NULL_HANDLE;
        sampler = VK_NULL_HANDLE;

        uploadTexture(path, allocator);
    }

    Image2D() {};

    void createDescriptors(uint32_t bindingIdx, VkShaderStageFlags flags = VK_SHADER_STAGE_FRAGMENT_BIT, VkDescriptorType descType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
        bindingIndex = bindingIdx;
        binding.binding = bindingIndex;
        binding.descriptorType = descType;
        binding.descriptorCount = 1;
        binding.stageFlags = flags;
        binding.pImmutableSamplers = nullptr;
    }

    // call after uploadTexture has setup the layout and assigned the memory
    void updateDescriptorSet(VkDescriptorSet& set, size_t arrayElement = 0) {
        imageInfo.imageLayout = imageLayout;
        imageInfo.imageView = imageView;
        imageInfo.sampler = sampler;

        wrt_desc_set.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wrt_desc_set.dstSet = set;
        wrt_desc_set.dstBinding = bindingIndex;
        wrt_desc_set.dstArrayElement = arrayElement;
        wrt_desc_set.descriptorCount = 1;
        wrt_desc_set.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        wrt_desc_set.pImageInfo = &imageInfo;
    }

    void generateMipmaps() {
        auto cmd = alloc->getSingleTimeCmd(true);
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.image = image;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;  // because only color images can have mipmaps. because why in the world would a depth image need mipaps.
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
        barrier.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;

        int32_t mipWidth = width;
        int32_t mipHeight = height;

        for (uint32_t i = 1; i < mipLevels; i++) {
            barrier.subresourceRange.baseMipLevel = i - 1;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 0, nullptr, 0, nullptr, 1, &barrier);

            VkImageBlit blit{};
            blit.srcOffsets[0] = { 0, 0, 0 };
            blit.srcOffsets[1] = { mipWidth, mipHeight, 1 };
            blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            blit.srcSubresource.mipLevel = i - 1;
            blit.srcSubresource.baseArrayLayer = 0;
            blit.srcSubresource.layerCount = 1;
            blit.dstOffsets[0] = { 0, 0, 0 };
            blit.dstOffsets[1] = { mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1 };
            blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            blit.dstSubresource.mipLevel = i;
            blit.dstSubresource.baseArrayLayer = 0;
            blit.dstSubresource.layerCount = 1;
            vkCmdBlitImage(cmd,
                image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1, &blit,
                VK_FILTER_LINEAR);

            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                0, nullptr,
                0, nullptr,
                1, &barrier);

            if (mipWidth > 1) mipWidth /= 2;
            if (mipHeight > 1) mipHeight /= 2;
        }

        barrier.subresourceRange.baseMipLevel = mipLevels - 1;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
            0, nullptr,
            0, nullptr,
            1, &barrier);

        alloc->init->disp.endCommandBuffer(cmd);
        alloc->submitSingleTimeCmd(cmd, false, true);
    }

    void uploadTexture(const std::string& path, Allocator* allocator) {

        alloc = allocator;

        int texWidth, texHeight, texChannels;
        stbi_uc* pixels = stbi_load(path.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        VkDeviceSize imageSize = texWidth * texHeight * 4;

        width = texWidth;
        height = texHeight;

        mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1;

        auto im = allocator->createImage(width, height, mipLevels, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_TYPE_2D, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_FORMAT_R8G8B8A8_SRGB, VK_SAMPLE_COUNT_1_BIT);
        this->image = im.first;
        this->memory = im.second;

        auto [buf, mem] = allocator->createBuffer(static_cast<VkDeviceSize>(width * height * 4), VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        stagingBuffer = buf;
        stagingMemory = mem;

        format = VK_FORMAT_R8G8B8A8_SRGB;
        usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        samples = VK_SAMPLE_COUNT_1_BIT;
        imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        createSampler();
        createImageView();

        void* data;
        allocator->init->disp.mapMemory(stagingMemory, 0, imageSize, 0, &data);
        std::memcpy(data, pixels, imageSize);
        allocator->init->disp.unmapMemory(stagingMemory);

        stbi_image_free(pixels);

        allocator->copyBufferToImage2D(stagingBuffer, image, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
        generateMipmaps(); // handles all the image layout transitions too
    }

    void createImageView(bool depthOnly = false) {
        // Create Image View
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format;
        if (!depthOnly) {
            viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        }
        else {
            viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        }

        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = mipLevels;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;
        viewInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        viewInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        viewInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        viewInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        if (alloc->init->disp.createImageView(&viewInfo, nullptr, &imageView) != VK_SUCCESS) {
            throw std::runtime_error("could not create image view");
        }
    }

    void createSampler() {
        // Create Sampler
        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.anisotropyEnable = VK_FALSE;
        samplerInfo.maxAnisotropy = 0;// alloc->init->device.physical_device.properties.limits.maxSamplerAnisotropy;;
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerInfo.mipLodBias = 0.0f;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = static_cast<float>(mipLevels);
        if (alloc->init->disp.createSampler(&samplerInfo, nullptr, &sampler) != VK_SUCCESS) {
            throw std::runtime_error("could not create sampler");
        }
    }

    std::unique_ptr<Image2D> deepcpy() {
        auto newImage = std::make_unique<Image2D>(width, height, mipLevels, format, usage, alloc, samples);
        newImage->imageLayout = imageLayout;
        newImage->bindingIndex = bindingIndex;
        newImage->createDescriptors(bindingIndex);
        return newImage;
    }
};

// uses bindless descriptors for Image2D access in shaders.
// ONE PIPELINE CAN HAVE ONLY ONE IMAGE ARRAY! (for now)
struct ImageArray {
    std::vector<Image2D> images;

    VkDescriptorSet descSet;
    VkDescriptorSetLayout descSetLayout;
    VkDescriptorPool descPool;

    uint32_t numImages = 100;

    Allocator* allocator;

    ImageArray(uint32_t numImages, Allocator* allocator) : numImages(numImages), allocator(allocator) {
        if (numImages == 0) {
            std::cerr << "numImages cannot be 0" << std::endl;
            return;
        }
        createDescriptorPool();
        createDescSetLayout();  // for now, the set = 1 is reserved for bindless textures via ImageArray
        allocateDescSet();
        allocator->init->addObject(this);
    };
    ImageArray() {};

    void destructor() {
        allocator->init->disp.destroyDescriptorPool(descPool, nullptr);
        allocator->init->disp.freeDescriptorSets(descPool, 1, &descSet);
        allocator->init->disp.destroyDescriptorSetLayout(descSetLayout, nullptr);
    }

    // creates new image and pushes it back into the array and descriptor set
    void push_back(const std::string& path) {
        images.emplace_back(path, allocator);
    }

    void erase(uint32_t index) {
        if (index >= images.size()) {
            throw std::out_of_range("Index out of range");
        }
        allocator->init->disp.destroyImageView(images[index].imageView, nullptr);
        allocator->init->disp.destroySampler(images[index].sampler, nullptr);
        allocator->init->disp.destroyImage(images[index].image, nullptr);
        allocator->init->disp.freeMemory(images[index].memory, nullptr);
        images.erase(images.begin() + index);
    }

    void createDescriptorPool() {
        VkDescriptorPoolSize pool_sizes_bindless[] =
        {
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, numImages }
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
        vk_binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        vk_binding.descriptorCount = numImages;
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
        uint32_t max_binding = 100 - 1;
        count_info.descriptorSetCount = 1;
        // This number is the max allocatable count
        count_info.pDescriptorCounts = &max_binding;
        alloc_info.pNext = &count_info;

        allocator->init->disp.allocateDescriptorSets(&alloc_info, &descSet);
    }

    void updateDescriptorSets() {
        std::vector<VkWriteDescriptorSet> writes(images.size());
        for (size_t i = 0; i < images.size(); ++i) {
            images[i].createDescriptors(0, VK_SHADER_STAGE_ALL);
            images[i].updateDescriptorSet(descSet, i);
            writes[i] = images[i].wrt_desc_set;
        }
        allocator->init->disp.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }
};

// a singular framebuffer attachment
struct FBAttachment {
    std::unique_ptr<Image2D> image;
    VkImageLayout initialLayout;
    VkImageLayout finalLayout;
    VkFormat format;

    VkAttachmentDescription attachmentDescription{};
    VkAttachmentReference attachmentReference{};

    FBAttachment() {};
    FBAttachment(int width, int height, int bindingIndex, VkFormat format, VkImageUsageFlags usage, VkImageLayout initialLayout, VkImageLayout finalLayout, VkImageLayout attachmentLayout, Allocator* allocator) : image(std::make_unique<Image2D>(width, height, 1, format, usage, allocator)), initialLayout(initialLayout), finalLayout(finalLayout), format(format) {

        image->createDescriptors(bindingIndex, VK_SHADER_STAGE_ALL);

        attachmentDescription.flags = 0;
        attachmentDescription.format = format;
        attachmentDescription.samples = VK_SAMPLE_COUNT_1_BIT;
        attachmentDescription.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachmentDescription.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachmentDescription.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachmentDescription.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachmentDescription.initialLayout = initialLayout;
        attachmentDescription.finalLayout = finalLayout;

        attachmentReference.attachment = bindingIndex;
        attachmentReference.layout = attachmentLayout;
    }

    std::shared_ptr<FBAttachment> deepcpy() {
        auto newAttachment = std::make_unique<FBAttachment>(image->width, image->height, attachmentReference.attachment, format, image->usage, initialLayout, finalLayout, attachmentReference.layout, image->alloc);
        return newAttachment;
    }
};
// a framebuffer object. Can be used for offscreen rendering, or as a render pass attachment
// This specific implementation now works for multiple framebuffers in flight, each with the same attachments.
struct Framebuffer {
    std::vector<VkFramebuffer> framebuffers;
    VkRenderPass renderPass;

    struct Attachments {
        std::vector<std::shared_ptr<FBAttachment>> attachments;
    };
    std::vector<Attachments> attachments;

    Allocator* allocator;

    VkSubpassDescription subpassDesc;
    VkSubpassDependency subpassDependency;

    int width, height, numAttachments, MAX_FRAMES;
    bool hasDepthStencil = false;

    Framebuffer() {};

    // MAX_FRAMES is used for multiple frames in flight. Generates MAX_FRAMES framebuffers, each with the same attachments. But of couese, the images are deep copied.
    Framebuffer(int width, int height, int numAttachments, int maxFrames, VkRenderPass renderPass, Allocator* allocator) : width(width), height(height), renderPass(renderPass), allocator(allocator), numAttachments(numAttachments), MAX_FRAMES(maxFrames) {
        attachments.resize(MAX_FRAMES);
        framebuffers.resize(MAX_FRAMES);
    }
    ~Framebuffer() {
        //allocator->init->disp.destroyFramebuffer(framebuffer, nullptr);
    }

    void addAttachment(VkFormat format, VkImageUsageFlags usage, VkImageLayout initialLayout, VkImageLayout attachmentLayout, VkImageLayout finalLayout) {
        int idx = attachments[0].attachments.size();
        for (int i = 0; i < MAX_FRAMES; i++) {
            attachments[i].attachments.push_back(std::make_shared<FBAttachment>(width, height, idx, format, usage, initialLayout, finalLayout, attachmentLayout, allocator));
        }
    }

    void addAttachment(std::shared_ptr<FBAttachment>& attachment) {
        auto it = std::find_if(attachments[0].attachments.begin(), attachments[0].attachments.end(),
            [&attachment](const std::shared_ptr<FBAttachment>& a) { return a->attachmentReference.attachment == attachment->attachmentReference.attachment; });
        if (it != attachments[0].attachments.end()) {
            return;
        }
        for (int i = 0; i < MAX_FRAMES; i++) {
            attachments[i].attachments.push_back(attachment->deepcpy());
        }
    }

    void init() {

        for (int i = 0; i < MAX_FRAMES; i++) {
            std::vector<VkImageView> views(numAttachments);
            for (int j = 0; j < numAttachments; j++) {
                views[j] = attachments[i].attachments[j]->image->imageView;
            }
            VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = views.size();
            framebufferInfo.pAttachments = views.data();
            framebufferInfo.width = width;
            framebufferInfo.height = height;
            framebufferInfo.layers = 1;
            if (allocator->init->disp.createFramebuffer(&framebufferInfo, nullptr, &framebuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create framebuffer!");
            }
            std::cout << "Framebuffer created with " << numAttachments << " attachments." << std::endl;
        }
    }

    std::vector<VkAttachmentDescription> getAttachmentDescriptions() {
        std::vector<VkAttachmentDescription> descs;
        // just take the first attachment set, because each set is identical.
        for (auto& attachment : attachments[0].attachments) {
            descs.push_back(attachment->attachmentDescription);
        }
        return descs;
    }

    std::vector<VkAttachmentReference> getAttachmentReferences() {
        std::vector<VkAttachmentReference> refs;
        for (auto& attachment : attachments[0].attachments) {
            refs.push_back(attachment->attachmentReference);
        }
        return refs;
    }
};

struct Swapchain {
    vkb::Swapchain swapchain;

    Init* init;

    int width, height;

    Swapchain() : init(nullptr) {};
    Swapchain(Init* init, int width, int height) :
        init(init), width(width), height(height) {
        createSwapchain();
    };

    void createSwapchain() {
        vkb::SwapchainBuilder builder{ init->device };

        VkSurfaceFormatKHR format{};
        format.format = VK_FORMAT_A2B10G10R10_UNORM_PACK32;
        format.colorSpace = VK_COLORSPACE_SRGB_NONLINEAR_KHR;

        VkPresentModeKHR presentMode = VK_PRESENT_MODE_MAILBOX_KHR;

        auto swap_ret = builder.set_desired_present_mode(presentMode)
            .set_desired_extent(width, height)
            .set_desired_format(format)
            .set_old_swapchain(swapchain)
            .build();

        if (!swap_ret) {
            std::cout << swap_ret.error().message() << " " << swap_ret.vk_result() << "\n";
        }

        swapchain = swap_ret.value();
    }
};

#pragma once  
#define GLM_FORCE_RADIANS  
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#define GLFW_INCLUDE_VULKAN
#define TINYOBJLOADER_IMPLEMENTATION
constexpr int MAX_FRAMES_IN_FLIGHT = 3;

#include <glm/glm.hpp>  
#include <glm/gtc/matrix_transform.hpp>  
#include <glm/gtc/type_ptr.hpp>  
#include <glm/gtx/hash.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>
#include <vulkan/vulkan_core.h>
#include <GLFW/glfw3.h>  
#include "VkBootstrap.h"  
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <unordered_map>
#include "VkMemAlloc.hpp"
#include "RendererBuilder.hpp"
#include "registry.hpp"
#include "tiny_obj_loader.h"

int calcium_device_initialization(Init* init, GLFWwindow* window, VkSurfaceKHR surface) {
    vkb::InstanceBuilder instance_builder;
    auto instance_ret = instance_builder
        .use_default_debug_messenger()
        .request_validation_layers()
        .require_api_version(VK_API_VERSION_1_3) // Important!  
        .build();
    if (!instance_ret) {
        std::cout << instance_ret.error().message() << "\n";
        return -1;
    }
    init->instance = instance_ret.value();
    init->inst_disp = init->instance.make_table();

    glfwCreateWindowSurface(init->instance.instance, window, nullptr, &surface);

    // Enable required features for bindless textures, buffer device address, shader objects, and ray tracing  
    VkPhysicalDeviceDescriptorIndexingFeaturesEXT descriptor_indexing_features = {};
    descriptor_indexing_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES_EXT;
    descriptor_indexing_features.descriptorBindingPartiallyBound = VK_TRUE;
    descriptor_indexing_features.descriptorBindingVariableDescriptorCount = VK_TRUE;
    descriptor_indexing_features.runtimeDescriptorArray = VK_TRUE;
    descriptor_indexing_features.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;

    VkPhysicalDeviceBufferDeviceAddressFeatures buffer_device_address_features = {};
    buffer_device_address_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
    buffer_device_address_features.bufferDeviceAddress = VK_TRUE;

    VkPhysicalDeviceShaderObjectFeaturesEXT shader_object_features = {};
    shader_object_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT;
    shader_object_features.shaderObject = VK_TRUE;

    VkPhysicalDeviceRayTracingPipelineFeaturesKHR ray_tracing_pipeline_features = {};
    ray_tracing_pipeline_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
    ray_tracing_pipeline_features.rayTracingPipeline = VK_TRUE;

    VkPhysicalDeviceAccelerationStructureFeaturesKHR acceleration_structure_features = {};
    acceleration_structure_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    acceleration_structure_features.accelerationStructure = VK_TRUE;

    VkPhysicalDeviceRayQueryFeaturesKHR ray_query_features = {};
    ray_query_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
    ray_query_features.rayQuery = VK_TRUE;

    // Chain the features together  
    ray_tracing_pipeline_features.pNext = &acceleration_structure_features;
    acceleration_structure_features.pNext = &ray_query_features;
    ray_query_features.pNext = &descriptor_indexing_features;
    descriptor_indexing_features.pNext = &buffer_device_address_features;
    buffer_device_address_features.pNext = &shader_object_features;

    vkb::PhysicalDeviceSelector phys_device_selector(init->instance);
    auto phys_device_ret = phys_device_selector
        .set_surface(surface)
        .add_required_extension(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME)
        .add_required_extension(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME)
        .add_required_extension(VK_EXT_SHADER_OBJECT_EXTENSION_NAME)
        .add_required_extension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME)
        .add_required_extension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)
        .add_required_extension(VK_KHR_RAY_QUERY_EXTENSION_NAME)
        .add_required_extension(VK_EXT_PIPELINE_CREATION_FEEDBACK_EXTENSION_NAME)
        .add_required_extension_features(descriptor_indexing_features)
        .add_required_extension_features(buffer_device_address_features)
        .add_required_extension_features(shader_object_features)
        .add_required_extension_features(ray_tracing_pipeline_features)
        .select();

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
    init->device = device_ret.value();
    init->disp = init->device.make_table();

    return 0;
}

// Read shader bytecode from a file
#ifndef readShaderCode
#define readShaderCode
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
#endif // !readShaderCode

struct VertexInputDesc {
    std::vector<VkVertexInputBindingDescription> bindings;
    std::vector<VkVertexInputAttributeDescription> attributes;

    VkPipelineVertexInputStateCreateFlags flags = 0;
};

struct Vertex {
    int positionIndex;
    int texCoordsIndex;
    int normalIndex;
    glm::vec3 tangent;
    glm::vec3 bitangent;
    int materialIndex;

    bool operator==(const Vertex& other) {
        return other.positionIndex == positionIndex && other.normalIndex == normalIndex && other.tangent == tangent && other.bitangent == bitangent && other.materialIndex == materialIndex;
    }

    static VertexInputDesc getVertexDesc() {
        VertexInputDesc desc;

        desc.bindings.push_back({
            0,                          // binding
            sizeof(Vertex),            // stride
            VK_VERTEX_INPUT_RATE_VERTEX // inputRate
            });

        desc.attributes = {
            {0, 0, VK_FORMAT_R32_SINT, offsetof(Vertex, positionIndex)},
            {1, 0, VK_FORMAT_R32_SINT, offsetof(Vertex, texCoordsIndex)},
            {2, 0, VK_FORMAT_R32_SINT, offsetof(Vertex, normalIndex)},
            {3, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, tangent)},
            {4, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, bitangent)},
            {5, 0, VK_FORMAT_R32_SINT, offsetof(Vertex, materialIndex)}
        };

        return desc;
    }
};

// Custom hash function for Vertex
namespace std {
    template <>
    struct hash<Vertex> {
        size_t operator()(const Vertex& vertex) const {
            size_t seed = 0;
            hash<float> hasher;

            // Has positionIndex
            seed ^= hasher(vertex.positionIndex) + 0x9e3779b9 + (seed << 6) + (seed >> 2);

            // Hash texCoordsIndex
            seed ^= hasher(vertex.texCoordsIndex) + 0x9e3779b9 + (seed << 6) + (seed >> 2);

            // Hash normalIndex
            seed ^= hasher(vertex.normalIndex) + 0x9e3779b9 + (seed << 6) + (seed >> 2);

            // Hash tangent
            seed ^= hasher(vertex.tangent.x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= hasher(vertex.tangent.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= hasher(vertex.tangent.z) + 0x9e3779b9 + (seed << 6) + (seed >> 2);

            // Hash bitangent
            seed ^= hasher(vertex.bitangent.x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= hasher(vertex.bitangent.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= hasher(vertex.bitangent.z) + 0x9e3779b9 + (seed << 6) + (seed >> 2);

            // Hash materialIndex
            seed ^= hasher(vertex.materialIndex) + 0x9e3779b9 + (seed << 6) + (seed >> 2);

            // Hash positionIndex
            seed ^= hasher(vertex.positionIndex) + 0x9e3779b9 + (seed << 6) + (seed >> 2);

            return seed;
        }
    };
}

struct Edge {
    Vertex* v1;
    Vertex* v2;

    Edge(Vertex* a, Vertex* b) {
        if (a < b) { v1 = a; v2 = b; }
        else { v1 = b; v2 = a; }
    }

    bool operator==(const Edge& other) const {
        return v1 == other.v1 && v2 == other.v2;
    }
};

struct Triangle {
    Vertex* v1;
    Vertex* v2;
    Vertex* v3;

    Triangle(Vertex* a, Vertex* b, Vertex* c) : v1(a), v2(b), v3(c) {}
};

// Custom hash function for Edge
namespace std {
    template <>
    struct hash<Edge> {
        size_t operator()(const Edge& edge) const {
            size_t seed = 0;
            hash<Vertex*> hasher;

            // Combine the hashes of both vertices (order-independent)
            seed ^= hasher(edge.v1) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= hasher(edge.v2) + 0x9e3779b9 + (seed << 6) + (seed >> 2);

            return seed;
        }
    };
}

// Custom hash function for Triangle
namespace std {
    template <>
    struct hash<Triangle> {
        size_t operator()(const Triangle& triangle) const {
            size_t seed = 0;
            hash<Vertex*> hasher;

            // Sort the pointers first (ensures order-independent hashing)
            std::array<Vertex*, 3> sortedVertices = { triangle.v1, triangle.v2, triangle.v3 };
            std::sort(sortedVertices.begin(), sortedVertices.end());

            // Combine hashes of all three vertices
            for (auto* v : sortedVertices) {
                seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }

            return seed;
        }
    };
}

// Define Edge comparison function
struct EdgeEqual {
    bool operator()(const Edge& e1, const Edge& e2) const {
        return (e1.v1 == e2.v1 && e1.v2 == e2.v2);
    }
};

// Define Triangle comparison function
struct TriangleEqual {
    bool operator()(const Triangle& t1, const Triangle& t2) const {
        std::array<Vertex*, 3> sortedT1 = { t1.v1, t1.v2, t1.v3 };
        std::array<Vertex*, 3> sortedT2 = { t2.v1, t2.v2, t2.v3 };

        std::sort(sortedT1.begin(), sortedT1.end());
        std::sort(sortedT2.begin(), sortedT2.end());

        return sortedT1 == sortedT2;
    }
};

struct PushConst {
    glm::mat4 model;
    VkDeviceAddress posBuf;
    VkDeviceAddress normalBuf;
    VkDeviceAddress uvBuf;
    VkDeviceAddress indexBuf;
    int materialIndex;
    int positionOffset;
    int uvOffset;
    int normalOffset;
    int indexOffset;
    int frameIndex = 0;
};

struct DepthPushConst {
    glm::mat4 model;
    VkDeviceAddress posBuf;
    VkDeviceAddress indexBuf;
    int positionOffset;
    int indexOffset;
};

// this allows easy subpass chaining
struct RenderSubpass {
    std::vector<FBAttachment*> colorAttachments;
    std::vector<FBAttachment*> inputAttachments;
    FBAttachment* depthAttachment;

    // Backing storage for VkAttachmentReferences (must persist)
    std::vector<VkAttachmentReference> inputAttachmentRefs{};
    std::vector<VkAttachmentReference> colorAttachmentRefs{};

    VkSubpassDescription subpassDescription{};
    VkSubpassDependency subpassDependency{};

    bool hasDepthStencil = false;

    // subpasses are a doubly linked list
    RenderSubpass* pNext = nullptr;
    RenderSubpass* pPrevious = nullptr;
    int idx = 0; // the index of this subpass in the linked list

    RenderSubpass() {};

    RenderSubpass(VkAccessFlags accessMask, VkPipelineStageFlags stageFlags, RenderSubpass* pPrevious) : pPrevious(pPrevious) {
        if (pPrevious == nullptr) {
            idx = 0;
            subpassDependency.srcSubpass = VK_SUBPASS_EXTERNAL;
            subpassDependency.dstSubpass = 0;
            subpassDependency.srcAccessMask = 0;
            subpassDependency.dstAccessMask = accessMask;
            subpassDependency.srcStageMask = stageFlags;
            subpassDependency.dstStageMask = stageFlags;
            subpassDependency.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT; // this is the default, but you can change it if needed
        }
        else if (pPrevious != nullptr) {
            pPrevious->pNext = this;
            idx = pPrevious->idx + 1;
            pPrevious->subpassDependency.dstSubpass = idx;
            subpassDependency.srcSubpass = idx - 1;
            subpassDependency.dstSubpass = idx;
            subpassDependency.srcAccessMask = pPrevious->subpassDependency.dstAccessMask;
            subpassDependency.dstAccessMask = accessMask;
            subpassDependency.srcStageMask = pPrevious->subpassDependency.dstStageMask;
            subpassDependency.dstStageMask = stageFlags;
            subpassDependency.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT; // this is the default, but you can change it if needed
        }
    };

    void addInputAttachment(FBAttachment* attachment) {
        inputAttachments.push_back(attachment);
        inputAttachmentRefs.resize(inputAttachments.size());
        for (size_t i = 0; i < inputAttachments.size(); ++i) {
            inputAttachmentRefs[i] = inputAttachments[i]->attachmentReference;
        }
        subpassDescription.inputAttachmentCount = inputAttachmentRefs.size();
        subpassDescription.pInputAttachments = inputAttachmentRefs.data();
    }

    void addColorAttachment(FBAttachment* attachment) {
        colorAttachments.push_back(attachment);
        colorAttachmentRefs.resize(colorAttachments.size());
        for (size_t i = 0; i < colorAttachments.size(); ++i) {
            colorAttachmentRefs[i] = colorAttachments[i]->attachmentReference;
        }
        subpassDescription.colorAttachmentCount = colorAttachmentRefs.size();
        subpassDescription.pColorAttachments = colorAttachmentRefs.data();
    }

    void addDepthStencilAttachment(FBAttachment* attachment) {
        depthAttachment = attachment;
        subpassDescription.pDepthStencilAttachment = &depthAttachment->attachmentReference;
    }

    std::vector<VkAttachmentReference> getColorAttachmentReferences() {
        std::vector<VkAttachmentReference> refs;
        for (auto& attachment : colorAttachments) {
            refs.push_back(attachment->attachmentReference);
        }
        return refs;
    }

    std::vector<VkAttachmentReference> getInputAttachmentReferences() {
        std::vector<VkAttachmentReference> refs;
        for (auto& attachment : inputAttachments) {
            refs.push_back(attachment->attachmentReference);
        }
        return refs;
    }
};

struct Pipeline {
    VkRenderPass renderPass;
    RenderSubpass subpass;
    VkPipeline pipeline;
    VkPipelineLayout layout;

    VkDescriptorSet descriptorSet;
    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorPool descriptorPool;

    std::vector<VkPushConstantRange> pushConsts;
    std::vector<std::unique_ptr<Image2D>> images;
    std::vector<ImageArray> imageArrays;
    std::vector<VkDescriptorSetLayout> descriptorSetLayouts;

    Framebuffer framebuffer;

    Swapchain* swapchain;
    Allocator* allocator;

    std::vector<VkShaderModule> shaders;
    std::vector<VkPipelineShaderStageCreateInfo> shader_stages;

    std::function<void()> createPipelineFunction;

    Pipeline(Swapchain* swapchain, Allocator* allocator, int width, int height) : swapchain(swapchain), allocator(allocator) {
        framebuffer = Framebuffer(width, height, 1, MAX_FRAMES_IN_FLIGHT, renderPass, allocator);
    }

    Pipeline(Swapchain* swapchain, Allocator* allocator) : swapchain(swapchain), allocator(allocator) {
        framebuffer = Framebuffer(swapchain->width, swapchain->height, 1, MAX_FRAMES_IN_FLIGHT, renderPass, allocator);
    }

    Pipeline() {};

    virtual void initialize() {

        if (framebuffer.attachments[0].attachments.size() > 0) {
            createRenderPassFB();
            framebuffer.renderPass = renderPass;
            framebuffer.init();
        }
        else {
            createRenderPassNoFB();
        }
        if (images.size() > 0) {
            createDescriptorPool();
            createDescriptorLayouts();
        }

        createPipelineLayout();
        createPipeline();
    }

    // no hand-holding here. You need to write this on your own.
    virtual void createPipeline() {
        createPipelineFunction();
    }

    void addImageArray(uint32_t maxImages) {
        ImageArray imageArray = ImageArray(maxImages, allocator);
        imageArray.updateDescriptorSets();
        imageArrays.push_back(imageArray);
    }

    void addPushConstant(VkDeviceSize range, VkDeviceSize offset, VkShaderStageFlags stage) {
        VkPushConstantRange constant = {};
        constant.offset = offset;
        constant.size = range;
        constant.stageFlags = stage;

        pushConsts.push_back(constant);
    }

    void addColorAttachment(std::shared_ptr<FBAttachment>& attachment) {
        subpass.addColorAttachment(attachment.get());
        framebuffer.addAttachment(attachment);
    }

    void setDepthAttachment(std::shared_ptr<FBAttachment>& attachment) {
        subpass.addDepthStencilAttachment(attachment.get());
        framebuffer.addAttachment(attachment);
    }

    void createDescriptorPool() {
        VkDescriptorPoolSize poolSizes = {
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            images.size()
        };

        VkDescriptorPoolCreateInfo poolInfo = {};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSizes;
        allocator->init->disp.createDescriptorPool(&poolInfo, nullptr, &descriptorPool);
    }

    void createDescriptorLayouts() {
        std::vector<VkDescriptorSetLayoutBinding> bindings(images.size());
        for (size_t i = 0; i < images.size(); ++i) {
            images[i]->createDescriptors(i, VK_SHADER_STAGE_ALL);
            bindings[i] = images[i]->binding;
        }
        VkDescriptorSetLayoutCreateInfo dsl_info = {};
        dsl_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dsl_info.bindingCount = images.size();
        dsl_info.pBindings = bindings.data();
        allocator->init->disp.createDescriptorSetLayout(&dsl_info, nullptr, &descriptorSetLayout);

        VkDescriptorSetAllocateInfo ds_allocate_info = {};
        ds_allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ds_allocate_info.descriptorPool = descriptorPool;
        ds_allocate_info.descriptorSetCount = 1;
        ds_allocate_info.pSetLayouts = &descriptorSetLayout;
        allocator->init->disp.allocateDescriptorSets(&ds_allocate_info, &descriptorSet);

        std::vector<VkWriteDescriptorSet> writeSets(images.size());
        for (size_t i = 0; i < images.size(); i++) {
            images[i]->updateDescriptorSet(descriptorSet, 0);
            writeSets[i] = images[i]->wrt_desc_set;
        }

        allocator->init->disp.updateDescriptorSets(images.size(), writeSets.data(), 0, nullptr);
        for (auto& imageArray : imageArrays) {
            imageArray.updateDescriptorSets();
        }
    }

    void addImage(std::unique_ptr<Image2D>& image) {
        image->createDescriptors(images.size(), VK_SHADER_STAGE_ALL);
        image->updateDescriptorSet(descriptorSet, 0);
        images.push_back(std::move(image));
    }

    // if there is custom framebuffer, that means it'll output to that framebuffer, which is handled by the pipeline object itself
    virtual void createRenderPassFB() {
        std::vector<VkAttachmentDescription> attachments = framebuffer.getAttachmentDescriptions();

        VkRenderPassCreateInfo render_pass_info = {};
        render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        render_pass_info.attachmentCount = attachments.size();
        render_pass_info.pAttachments = attachments.data();
        render_pass_info.subpassCount = 1;
        render_pass_info.pSubpasses = &subpass.subpassDescription;
        render_pass_info.dependencyCount = 1;
        render_pass_info.pDependencies = &subpass.subpassDependency;
        render_pass_info.flags = 0;
        render_pass_info.pNext = nullptr;

        if (allocator->init->disp.createRenderPass(&render_pass_info, nullptr, &renderPass) != VK_SUCCESS) {
            std::cout << "failed to create render pass\n";
        }
    }

    // no frameBuffer attached means this is the final, present pipeline. It'll output to the swapchain's framebuffers, which is handled by the engine.
    virtual void createRenderPassNoFB() {
        VkAttachmentDescription color_attachment = {};
        color_attachment.format = swapchain->swapchain.image_format;
        color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
        color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference color_attachment_ref = {};
        color_attachment_ref.attachment = 0;
        color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass = {};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &color_attachment_ref;

        VkSubpassDependency dependency = {};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        VkRenderPassCreateInfo render_pass_info = {};
        render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        render_pass_info.attachmentCount = 1;
        render_pass_info.pAttachments = &color_attachment;
        render_pass_info.subpassCount = 1;
        render_pass_info.pSubpasses = &subpass;
        render_pass_info.dependencyCount = 1;
        render_pass_info.pDependencies = &dependency;

        if (allocator->init->disp.createRenderPass(&render_pass_info, nullptr, &renderPass) != VK_SUCCESS) {
            std::cout << "failed to create render pass\n";
        }
    }

    virtual void createPipelineLayout() {

        VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;

        if (images.size() > 0 && imageArrays.size() > 0) {
            descriptorSetLayouts.resize(imageArrays.size() + 1);
            descriptorSetLayouts[0] = descriptorSetLayout;
            for (size_t i = 0; i < imageArrays.size(); ++i) {
                descriptorSetLayouts[i + 1] = imageArrays[i].descSetLayout;
            }
            pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(descriptorSetLayouts.size());
            pipelineLayoutInfo.pSetLayouts = descriptorSetLayouts.data();
        }
        else if (images.size() == 0 && imageArrays.size() > 0) {
            descriptorSetLayouts.resize(imageArrays.size());
            for (size_t i = 0; i < imageArrays.size(); ++i) {
                descriptorSetLayouts[i] = imageArrays[i].descSetLayout;
            }
            pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(descriptorSetLayouts.size());
            pipelineLayoutInfo.pSetLayouts = descriptorSetLayouts.data();
        }
        else if (images.size() > 0 && imageArrays.size() == 0) {
            pipelineLayoutInfo.setLayoutCount = 1;
            pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
        }
        else {
            pipelineLayoutInfo.setLayoutCount = 0;
            pipelineLayoutInfo.pSetLayouts = nullptr;
        }

        pipelineLayoutInfo.pushConstantRangeCount = pushConsts.size();
        pipelineLayoutInfo.pPushConstantRanges = pushConsts.data();

        if (allocator->init->disp.createPipelineLayout(&pipelineLayoutInfo, nullptr, &layout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }
    }

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

    void addShaderModule(const std::string& path, VkShaderStageFlagBits stage) {
        auto code = readShaderBytecode(path);
        VkShaderModuleCreateInfo create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        create_info.codeSize = code.size();
        create_info.pCode = reinterpret_cast<const uint32_t*>(code.data());

        VkShaderModule shaderModule;
        if (allocator->init->disp.createShaderModule(&create_info, nullptr, &shaderModule) != VK_SUCCESS) {
            throw std::exception("couldn't add shader module");
        }

        VkPipelineShaderStageCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        info.stage = stage;
        info.module = shaderModule;
        info.pName = "main";

        shaders.push_back(shaderModule);
        shader_stages.push_back(info);
    }

    void bindDescSets(VkCommandBuffer& cmd) const {
        if (images.size() > 0) {
            allocator->init->disp.cmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &descriptorSet, 0, nullptr);
            for (int i = 0; i < imageArrays.size(); i++) {
                allocator->init->disp.cmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, i + 1, 1, &imageArrays[i].descSet, 0, nullptr);
            }
        }
        else if (images.size() == 0 && imageArrays.size() > 0) {
            for (int i = 0; i < imageArrays.size(); i++) {
                allocator->init->disp.cmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, i, 1, &imageArrays[i].descSet, 0, nullptr);
            }
        }
    }

    void render(VkCommandBuffer& cmd, VkCommandBuffer& renderCmds, VkClearValue& clearValue, int frameIndex) const {

        VkRenderPassBeginInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        info.renderPass = renderPass;
        info.framebuffer = framebuffer.framebuffers[frameIndex];
        info.renderArea.offset = { 0, 0 };
        info.renderArea.extent = swapchain->swapchain.extent;
        info.clearValueCount = 1;
        info.pClearValues = &clearValue;

        allocator->init->disp.cmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        if (images.size() > 0 || imageArrays.size() > 0) { bindDescSets(cmd); }
        allocator->init->disp.cmdBeginRenderPass(cmd, &info, VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS);
        allocator->init->disp.cmdExecuteCommands(cmd, 1, &renderCmds);
        allocator->init->disp.cmdEndRenderPass(cmd);
    }

    void render(VkCommandBuffer& cmd, VkCommandBuffer& renderCmds, VkRenderPassBeginInfo& info) {
        allocator->init->disp.cmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        if (images.size() > 0 || imageArrays.size() > 0) { bindDescSets(cmd); }
        allocator->init->disp.cmdBeginRenderPass(cmd, &info, VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS);
        allocator->init->disp.cmdExecuteCommands(cmd, 1, &renderCmds);
        allocator->init->disp.cmdEndRenderPass(cmd);
    }
};

struct DepthPipeline : public Pipeline {

    DepthPipeline() : Pipeline() {};

    DepthPipeline(Swapchain* swapchain, Allocator* allocator) : Pipeline(swapchain, allocator) {};

    DepthPipeline(const std::string& vertexPath, const std::string& fragmentPath, Swapchain* swapchain, Allocator* allocator) : Pipeline(swapchain, allocator) {
        addShaderModule(vertexPath, VK_SHADER_STAGE_VERTEX_BIT);
        addShaderModule(fragmentPath, VK_SHADER_STAGE_FRAGMENT_BIT);
        addPushConstant(sizeof(DepthPushConst), 0, VK_SHADER_STAGE_VERTEX_BIT);

        subpass = RenderSubpass(VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, nullptr);

        auto depthAttachment = std::make_shared<FBAttachment>(swapchain->width, swapchain->height, 0, VK_FORMAT_D32_SFLOAT,
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, allocator);

        setDepthAttachment(depthAttachment);

        initialize();

        //allocator->init->addObject(this);
    }

    void createPipeline() override {

        VkGraphicsPipelineCreateInfo pipelineInfo = {};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = shaders.size();

        // No vertex input state needed because we use a custom vertex processor
        VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 0; // No bindings
        vertexInputInfo.pVertexBindingDescriptions = nullptr; // No binding descriptions
        vertexInputInfo.vertexAttributeDescriptionCount = 0; // No attributes
        vertexInputInfo.pVertexAttributeDescriptions = nullptr; // No attribute descriptions

        pipelineInfo.pVertexInputState = &vertexInputInfo;

        // Input Assembly
        VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;
        pipelineInfo.pInputAssemblyState = &inputAssembly;

        VkViewport viewport = {};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float)swapchain->width;
        viewport.height = (float)swapchain->height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        VkRect2D scissor = {};
        scissor.offset = { 0, 0 };
        scissor.extent = swapchain->swapchain.extent;

        VkPipelineViewportStateCreateInfo viewport_state = {};
        viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewport_state.viewportCount = 1;
        viewport_state.pViewports = &viewport;
        viewport_state.scissorCount = 1;
        viewport_state.pScissors = &scissor;

        VkPipelineRasterizationStateCreateInfo rasterizer = {};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;

        VkPipelineMultisampleStateCreateInfo multisampling = {};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
        colorBlendAttachment.colorWriteMask = 0; // Disable all color writes
        colorBlendAttachment.blendEnable = VK_FALSE; // No blending needed

        VkPipelineColorBlendStateCreateInfo color_blending = {};
        color_blending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        color_blending.logicOpEnable = VK_FALSE; // Logic operations not needed
        color_blending.logicOp = VK_LOGIC_OP_COPY; // Default logic operation
        color_blending.attachmentCount = 1; // One attachment
        color_blending.pAttachments = &colorBlendAttachment;
        color_blending.blendConstants[0] = 0.0f;
        color_blending.blendConstants[1] = 0.0f;
        color_blending.blendConstants[2] = 0.0f;
        color_blending.blendConstants[3] = 0.0f;

        std::vector<VkDynamicState> dynamic_states = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };

        VkPipelineDynamicStateCreateInfo dynamic_info = {};
        dynamic_info.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamic_info.dynamicStateCount = static_cast<uint32_t>(dynamic_states.size());
        dynamic_info.pDynamicStates = dynamic_states.data();

        VkPipelineDepthStencilStateCreateInfo depthStencil = {};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_TRUE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_GREATER; // REVERSE-Z: greater is closer
        depthStencil.depthBoundsTestEnable = VK_FALSE;
        depthStencil.minDepthBounds = 0.0f;
        depthStencil.maxDepthBounds = 1.0f;
        depthStencil.stencilTestEnable = VK_FALSE;
        depthStencil.front = {};
        depthStencil.back = {};

        VkGraphicsPipelineCreateInfo pipeline_info = {};
        pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipeline_info.stageCount = shader_stages.size();
        pipeline_info.pStages = shader_stages.data();
        pipeline_info.pVertexInputState = &vertexInputInfo;
        pipeline_info.pInputAssemblyState = &inputAssembly;
        pipeline_info.pViewportState = &viewport_state;
        pipeline_info.pRasterizationState = &rasterizer;
        pipeline_info.pMultisampleState = &multisampling;
        pipeline_info.pColorBlendState = &color_blending;
        pipeline_info.pDynamicState = &dynamic_info;
        pipeline_info.pDepthStencilState = &depthStencil;
        pipeline_info.layout = layout;
        pipeline_info.renderPass = renderPass;
        pipeline_info.subpass = 0;
        pipeline_info.basePipelineHandle = VK_NULL_HANDLE;

        if (allocator->init->disp.createGraphicsPipelines(VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline) != VK_SUCCESS) {
            throw std::exception("failed to create graphics pipeline");
        }

        for (auto& shader : shaders) {
            allocator->init->disp.destroyShaderModule(shader, nullptr);
        }
    }
};

struct GraphicsPipeline : public Pipeline {

    Init* init;
    int width, height;

    std::vector<std::unique_ptr<Pipeline>> pipelines;

    GraphicsPipeline(const std::string& vertexPath, const std::string& fragmentPath, Init* init, Swapchain* swapchain, Allocator* allocator, int width, int height) : width(width), height(height), init(init), Pipeline(swapchain, allocator) {

        addShaderModule(vertexPath, VK_SHADER_STAGE_VERTEX_BIT);
        addShaderModule(fragmentPath, VK_SHADER_STAGE_FRAGMENT_BIT);
        addPushConstant(sizeof(PushConst), 0, VK_SHADER_STAGE_VERTEX_BIT);

        std::unique_ptr<Pipeline> depthPipeline = std::make_unique<DepthPipeline>("depth.vert.spv", "depth.frag.spv", swapchain, allocator);
        addPipeline(depthPipeline);

        addImage(pipelines[0]->framebuffer.attachments[0].attachments[0]->image); // add depth image to this pipeline (in this case, it's set = 0, binding = 0)
        addImage(pipelines[0]->framebuffer.attachments[1].attachments[0]->image); // add depth image number 2 to this pipeline (in this case, it's set = 0, binding = 1)
        addImage(pipelines[0]->framebuffer.attachments[2].attachments[0]->image); // add depth image number 3 to this pipeline (in this case, it's set = 0, binding = 2)
        addImageArray(100);
        initialize();

        //init->addObject(this);
    }

    GraphicsPipeline(Init* init, Swapchain* swapchain, Allocator* allocator) : init(init), Pipeline(swapchain, allocator) {}

    GraphicsPipeline() {};

    // assumes that the pipeline is already initialized
    void addPipeline(std::unique_ptr<Pipeline>& pipeline) {
        pipelines.push_back(std::move(pipeline));
    }

    void createPipeline() override {
        VkGraphicsPipelineCreateInfo pipelineInfo = {};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = shaders.size();

        std::vector<VkPipelineShaderStageCreateInfo> shaderStages(shaders.size());
        for (size_t i = 0; i < shaders.size(); ++i) {
            shaderStages[i] = shader_stages[i];
        }
        pipelineInfo.pStages = shaderStages.data();

        // No vertex input state needed
        VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 0; // No bindings
        vertexInputInfo.pVertexBindingDescriptions = nullptr; // No binding descriptions
        vertexInputInfo.vertexAttributeDescriptionCount = 0; // No attributes
        vertexInputInfo.pVertexAttributeDescriptions = nullptr; // No attribute descriptions

        pipelineInfo.pVertexInputState = &vertexInputInfo;

        // Input Assembly
        VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;
        pipelineInfo.pInputAssemblyState = &inputAssembly;

        VkViewport viewport = {};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float)swapchain->width;
        viewport.height = (float)swapchain->height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        VkRect2D scissor = {};
        scissor.offset = { 0, 0 };
        scissor.extent = swapchain->swapchain.extent;

        VkPipelineViewportStateCreateInfo viewport_state = {};
        viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewport_state.viewportCount = 1;
        viewport_state.pViewports = &viewport;
        viewport_state.scissorCount = 1;
        viewport_state.pScissors = &scissor;

        VkPipelineRasterizationStateCreateInfo rasterizer = {};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;

        VkPipelineMultisampleStateCreateInfo multisampling = {};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
        colorBlendAttachment.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo color_blending = {};
        color_blending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        color_blending.logicOpEnable = VK_FALSE;
        color_blending.logicOp = VK_LOGIC_OP_COPY;
        color_blending.attachmentCount = 1;
        color_blending.pAttachments = &colorBlendAttachment;
        color_blending.blendConstants[0] = 0.0f;
        color_blending.blendConstants[1] = 0.0f;
        color_blending.blendConstants[2] = 0.0f;
        color_blending.blendConstants[3] = 0.0f;

        std::vector<VkDynamicState> dynamic_states = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };

        VkPipelineDynamicStateCreateInfo dynamic_info = {};
        dynamic_info.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamic_info.dynamicStateCount = static_cast<uint32_t>(dynamic_states.size());
        dynamic_info.pDynamicStates = dynamic_states.data();

        VkGraphicsPipelineCreateInfo pipeline_info = {};
        pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipeline_info.stageCount = shaderStages.size();
        pipeline_info.pStages = shaderStages.data();
        pipeline_info.pVertexInputState = &vertexInputInfo;
        pipeline_info.pInputAssemblyState = &inputAssembly;
        pipeline_info.pViewportState = &viewport_state;
        pipeline_info.pRasterizationState = &rasterizer;
        pipeline_info.pMultisampleState = &multisampling;
        pipeline_info.pColorBlendState = &color_blending;
        pipeline_info.pDynamicState = &dynamic_info;
        pipeline_info.layout = layout;
        pipeline_info.renderPass = renderPass;
        pipeline_info.subpass = 0;
        pipeline_info.basePipelineHandle = VK_NULL_HANDLE;

        if (init->disp.createGraphicsPipelines(VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline) != VK_SUCCESS) {
            std::cout << "failed to create pipline\n";
        }

        for (auto& shader : shaders) {
            init->disp.destroyShaderModule(shader, nullptr);
        }
    }
};

struct Material {
    std::string texPath;

    Material() {};

    void loadTexture(const std::string& path) {
        texPath = path;
    }
};

struct PackedIndex {
    uint32_t pos;
    uint32_t norm;
    uint32_t uv;
    uint32_t _padding;
};

struct Mesh {
    std::vector<Triangle> triangles;
    std::vector<glm::vec4> positions;
    std::vector<glm::vec2> uvs;
    std::vector<glm::vec4> normals;
    std::vector<PackedIndex> packedIndices;

    unsigned int indexCount;

    Material material;

    bool meshUploaded = false;

    Mesh() {};

    void loadModel(const std::string& objFilePath) {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn, err;

        bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, objFilePath.c_str());

        if (!warn.empty()) std::cerr << "[WARN] " << warn << std::endl;
        if (!err.empty()) std::cerr << "[ERROR] " << err << std::endl;
        if (!ret) return;

        // Convert positions
        for (size_t i = 0; i < attrib.vertices.size(); i += 3) {
            glm::vec4 pos(attrib.vertices[i], attrib.vertices[i + 1], attrib.vertices[i + 2], 0.0f);
            positions.push_back(pos);
        }

        // Convert uvs
        for (size_t i = 0; i < attrib.texcoords.size(); i += 2) {
            glm::vec2 uv(attrib.texcoords[i], attrib.texcoords[i + 1]);
            uvs.push_back(uv);
        }

        // Convert normals
        for (size_t i = 0; i < attrib.normals.size(); i += 3) {
            glm::vec4 norm(attrib.normals[i], attrib.normals[i + 1], attrib.normals[i + 2], 0.0f);
            normals.push_back(norm);
        }

        // Process shapes
        for (const auto& shape : shapes) {
            for (const auto& index : shape.mesh.indices) {
                packedIndices.push_back({
                    static_cast<uint32_t>(index.vertex_index),
                    static_cast<uint32_t>(index.normal_index),
                    static_cast<uint32_t>(index.texcoord_index)
                    });
            }
        }
    }

    Mesh(const Mesh& other) {
        material = other.material;
        meshUploaded = other.meshUploaded;
    }

    ~Mesh() {}
};

struct PointLight {
    glm::vec3 position;
    glm::vec3 color;
    float intensity;
    PointLight(glm::vec3 position, glm::vec3 color, float intensity) : position(position), color(color), intensity(intensity) {}
};

enum Camera_Movement {
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT
};

class Camera {
    glm::vec3 Up;
    glm::vec3 Right;
    glm::vec3 WorldUp;
    float Yaw;
    float Pitch;

    float MovementSpeed;
    float MouseSensitivity;

public:

    glm::vec3 Front;
    glm::vec3 Position;
    float Zoom;
    Camera(glm::vec3 position, glm::vec3 up, float yaw, float pitch) : Front(glm::vec3(0.0f, 0.0f, -1.0f)), MovementSpeed(25.5f), MouseSensitivity(0.1f), Zoom(45.0f) {
        Position = position;
        WorldUp = up;
        Yaw = yaw;
        Pitch = pitch;
        updateCameraVectors();
    }

    Camera() {};

    glm::mat4 GetViewMatrix() const {
        return glm::lookAt(Position, Position + Front, Up);
    }

    glm::mat4 GetProjectionMatrix(float windowHeight, float windowWidth, float FOV) {
        return glm::perspective(glm::radians(FOV), windowWidth / windowHeight, 1000.0f, 1.0f);
    }

    glm::mat4 GetProjectionMatrixReverse(float windowHeight, float windowWidth, float FOV) {
        glm::mat4 projectionMatrix = glm::perspective(glm::radians(FOV), windowWidth / windowHeight, 1000.0f, 1.0f);
        return projectionMatrix;
    }

    glm::mat4 GetOrthogonalMatrix(float windowHeight, float windowWidth, float FOV) {
        float aspectRatio = windowWidth / windowHeight;
        float orthoHeight = FOV;
        float orthoWidth = FOV * aspectRatio;
        return glm::ortho<float>(-120, 120, -120, 120, -500, 500);
    }

    void ProcessKeyboard(int direction, float deltaTime) {
        float velocity = MovementSpeed * deltaTime;
        if (direction == FORWARD)
            Position += Front * velocity;
        if (direction == BACKWARD)
            Position -= Front * velocity;
        if (direction == LEFT)
            Position -= Right * velocity;
        if (direction == RIGHT)
            Position += Right * velocity;
    }

    void ProcessMouseMovement(float xoffset, float yoffset, GLboolean constrainPitch) {
        xoffset *= MouseSensitivity;
        yoffset *= MouseSensitivity;

        Yaw += xoffset;
        Pitch += yoffset;

        xoffset = 0.0f;
        yoffset = 0.0f;

        if (constrainPitch) {
            if (Pitch > 89.0f)
                Pitch = 89.0f;
            if (Pitch < -89.0f)
                Pitch = -89.0f;
        }

        updateCameraVectors();
    }

    void ProcessMouseScroll(float yoffset) {
        Zoom -= yoffset;
        if (Zoom < 1.0f)
            Zoom = 1.0f;
        if (Zoom > 45.0f)
            Zoom = 45.0f;
    }

    void updateCameraVectors() {
        glm::vec3 front;
        front.x = cos(glm::radians(Yaw)) * cos(glm::radians(Pitch));
        front.y = sin(glm::radians(Pitch));
        front.z = sin(glm::radians(Yaw)) * cos(glm::radians(Pitch));
        Front = glm::normalize(front);

        Right = glm::normalize(glm::cross(Front, WorldUp));
        Up = glm::normalize(glm::cross(Right, Front));
    }
};

struct Entity;

struct Transform {

    glm::vec3 position;
    glm::quat rotation;
    glm::vec3 scale;

    glm::mat4 model;

    Transform(glm::vec3 position, glm::quat rotation, glm::vec3 scale) : position(position), rotation(rotation), scale(scale) {}

    void updateGlobalTransform() {
        glm::mat4 localTransform = glm::mat4(1.0f);
        localTransform = glm::translate(localTransform, position);
        localTransform *= glm::toMat4(rotation);
        localTransform = glm::scale(localTransform, scale);
        model = localTransform;
    }

    glm::vec3 getPosition() const {
        return position;
    }
    glm::quat getRotation() const {
        return rotation;
    }
    glm::vec3 getScale() const {
        return scale;
    }

    void setPosition(glm::vec3 position) {
        this->position = position;
    }
    void setRotation(glm::quat rotation) {
        this->rotation = rotation;
    }
    void setScale(glm::vec3 scale) {
        this->scale = scale;
    }

    void translate(glm::vec3 translation) {
        position += translation;
    }
    void rotate(glm::quat rotation) {
        this->rotation = rotation * this->rotation;
    }
    void scaleBy(glm::vec3 scale) {
        this->scale *= scale;
    }
};

struct Entity {
    Transform transform;
    Mesh mesh;
    Init* init;

    Entity(glm::vec3 position, glm::quat rotation, glm::vec3 scale, bool useBuiltInTransform)
        : transform(position, rotation, scale) {
        //mesh.traingle();
    }

    void update() { transform.updateGlobalTransform(); }

private:

};

struct Scene {
    std::vector<Entity> entities;
    Camera camera;

    Allocator* allocator;

    // non duplicated unique vertex data
    // Vec4 instead of Vec3 because Vec4 is perfectly sized at 16 bytes, which means that std430 inside of shaders will work (because vec3's stride is 16 bytes).
    // Even though we're wasting 4 bytes per attribute, this is way faster than using Vec3, which is 12 bytes and is slow.
    // Buffer_reference_align is always set to 16 bytes, because that seems to be minimum vulkan alignment, according to vkGetPhysicalDeviceProperties
    // You can get the min alignment of ur device from Allocator->getAlignment()

    MemPool<glm::vec4> positionPool;
    MemPool<glm::vec2> uvPool;
    MemPool<glm::vec4> normalPool;

    // index data for all meshes. Avoids duplicated vertex attributes.
    // The number of elements inside of one mesh's index buffer is equal to the number of times the vertex shader will be called for that mesh
    MemPool<PackedIndex> packedIndexPool;

    GraphicsPipeline* pipeline;

    Scene(Allocator* allocator, GraphicsPipeline* pipeline) : allocator(allocator), pipeline(pipeline),
        positionPool(MemPool<glm::vec4>(1000, allocator, VK_SHADER_STAGE_VERTEX_BIT)),
        uvPool(MemPool<glm::vec2>(1000, allocator, VK_SHADER_STAGE_VERTEX_BIT)),
        normalPool(MemPool<glm::vec4>(1000, allocator, VK_SHADER_STAGE_VERTEX_BIT)),
        packedIndexPool(MemPool<PackedIndex>(1000, allocator, VK_SHADER_STAGE_VERTEX_BIT)) {

        entities.reserve(20);
        camera = Camera(glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), 0.0f, 0.0f);
    };

    void addEntity(Entity& entity) {
        entities.push_back(entity);
        positionPool.push_back(entity.mesh.positions);
        uvPool.push_back(entity.mesh.uvs);
        normalPool.push_back(entity.mesh.normals);
        packedIndexPool.push_back(entity.mesh.packedIndices);
        pipeline->imageArrays[0].push_back(entity.mesh.material.texPath);
        pipeline->imageArrays[0].updateDescriptorSets();
    }

    void removeEntity(uint32_t idx, bool instaClean = true) {
        if (idx >= entities.size()) return;
        //entities.erase(entities.begin() + idx);
        //positionPool.erase(idx, instaClean);
        //uvPool.erase(idx, instaClean);
        //normalPool.erase(idx, instaClean);
        //packedIndexPool.erase(idx, instaClean);
        //pipeline->textures.erase(idx);
        //pipeline->textures.updateDescriptorSets();
    }

    void defragment() {
        positionPool.cleanGaps();
        uvPool.cleanGaps();
        normalPool.cleanGaps();
        packedIndexPool.cleanGaps();
    }

    void renderScene(VkCommandBuffer& commandBuffer, int width, int height, int frameIndex) {
        PushConst pushConst = {};
        pushConst.posBuf = positionPool.getBufferAddress();
        pushConst.normalBuf = normalPool.getBufferAddress();
        pushConst.uvBuf = uvPool.getBufferAddress();
        pushConst.indexBuf = packedIndexPool.getBufferAddress();
        pushConst.frameIndex = frameIndex;

        for (int i = 0; i < entities.size(); i++) {
            auto& entity = entities[i];
            entity.update();
            pushConst.model = camera.GetProjectionMatrix(height, width, 50.0f) * camera.GetViewMatrix() * entity.transform.model;
            pushConst.materialIndex = i;

            // element offsets represent the beginning of the current mesh's data in the shared memPools
            pushConst.positionOffset = positionPool.buffers[i].elementOffset;
            pushConst.uvOffset = uvPool.buffers[i].elementOffset;
            pushConst.normalOffset = normalPool.buffers[i].elementOffset;
            pushConst.indexOffset = packedIndexPool.buffers[i].elementOffset;

            allocator->init->disp.cmdPushConstants(commandBuffer, pipeline->layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConst), &pushConst);
            allocator->init->disp.cmdDraw(commandBuffer, packedIndexPool.buffers[i].numElements, 1, 0, 0);
        }
    };

    void renderSceneDepth(VkCommandBuffer& commandBuffer, int width, uint32_t height) {
        DepthPushConst pushConst = {};
        pushConst.indexBuf = packedIndexPool.getBufferAddress();
        pushConst.posBuf = positionPool.getBufferAddress();

        for (int i = 0; i < entities.size(); i++) {
            auto& entity = entities[i];
            entity.update();
            pushConst.model = camera.GetProjectionMatrixReverse(height, width, 50.0f) * camera.GetViewMatrix() * entity.transform.model;
            pushConst.indexOffset = packedIndexPool.buffers[i].elementOffset;
            pushConst.positionOffset = positionPool.buffers[i].elementOffset;

            allocator->init->disp.cmdPushConstants(commandBuffer, pipeline->pipelines[0]->layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(DepthPushConst), &pushConst);
            allocator->init->disp.cmdDraw(commandBuffer, packedIndexPool.buffers[i].numElements, 1, 0, 0);
        }
    }

    void renderDirectionalShadows(VkCommandBuffer& commandBuffer, int size) {
        DepthPushConst pushConst = {};
        pushConst.indexBuf = packedIndexPool.getBufferAddress();
        pushConst.posBuf = positionPool.getBufferAddress();
        glm::mat4 lightProj = glm::ortho<float>(-size, size, -size, size, 1000.0f, 0.1f);
        glm::vec3 lightPos = camera.Position - glm::normalize(glm::vec3(45.0f, 45.0f, 45.0f)) * 100.0f;
        glm::mat4 lightView = glm::lookAt(lightPos, lightPos + glm::normalize(glm::vec3(45.0f, 45.0f, 45.0f)), glm::vec3(0.0f, 1.0f, 0.0f));

        for (int i = 0; i < entities.size(); i++) {
            auto& entity = entities[i];
            entity.update();
            pushConst.model = lightProj * lightView * entity.transform.model;
            pushConst.indexOffset = packedIndexPool.buffers[i].elementOffset;
            pushConst.positionOffset = positionPool.buffers[i].elementOffset;
            allocator->init->disp.cmdPushConstants(commandBuffer, pipeline->pipelines[0]->layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(DepthPushConst), &pushConst);
            allocator->init->disp.cmdDraw(commandBuffer, packedIndexPool.buffers[i].numElements, 1, 0, 0);
        }
    }

    Scene() {};
};

struct Engine {

    Init init;

    GLFWwindow* window;

    VkSurfaceKHR surface;

    Allocator* allocator;

    VkQueue graphicsQueue;
    VkQueue presentQueue;

    Swapchain swapchain;
    GraphicsPipeline pipeline;

    std::vector<VkImage> swapchainImages;
    std::vector<VkImageView> swapchainImageViews;
    std::vector<VkFramebuffer> framebuffers;

    VkCommandPool commandPool;
    std::vector<VkCommandBuffer> commandBuffers;
    std::vector<VkCommandBuffer> secondaryCmdBufs;
    std::vector<VkCommandBuffer> secondaryCmdBufsDepth;

    std::vector<VkSemaphore> availiables;
    std::vector<VkSemaphore> finishes;
    std::vector<VkFence> inFlights;
    std::vector<VkFence> imagesInFlights;

    size_t currentFrame = 0;

    int width, height;

    static Scene scene;

    Engine(int width, int height, const std::string& vertexPath, const std::string& fragmentPath) : height(height), width(width) {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        window = glfwCreateWindow(width, height, "Calcium", nullptr, nullptr);
        if (!window) {
            throw std::runtime_error("failed to create window");
        }

        glfwSetCursorPosCallback(window, mouse_callback);
        glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

        calcium_device_initialization(&init, window, surface);

        allocator = new Allocator(&init);

        createSwapchain();
        get_queues();
        createCommandPool();

        allocator->graphicsPool = commandPool;
        allocator->graphicsQueue = graphicsQueue;

        createGraphicsPipeline(vertexPath, fragmentPath);

        scene = Scene(allocator, &pipeline);

        auto entity = new Entity(glm::vec3(1.0f), glm::quat(1.0f, 0.0f, 0.0f, 0.0f), glm::vec3(1.0f), true);
        entity->mesh.loadModel("cube.obj");
        entity->mesh.material.loadTexture("test.jpg");

        auto entity2 = new Entity(glm::vec3(0.0f, -1.0f, 0.0f), glm::quat(1.0f, 0.0f, 0.0f, 0.0f), glm::vec3(10.0f, 10.0f, 10.0f), true);
        entity2->mesh.loadModel("bunny.obj");
        entity2->mesh.material.loadTexture("bricks.jpg");

        scene.addEntity(*entity);
        scene.addEntity(*entity2);

        createFramebuffers();
        createCommandBuffers();
        createSyncObjects();
    };

    ~Engine() {
        init.destroy();
        delete allocator;
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            init.disp.destroySemaphore(finishes[i], nullptr);
            init.disp.destroySemaphore(availiables[i], nullptr);
            init.disp.destroyFence(inFlights[i], nullptr);
        }

        init.disp.destroyCommandPool(commandPool, nullptr);

        for (auto framebuffer : framebuffers) {
            init.disp.destroyFramebuffer(framebuffer, nullptr);
        }

        init.disp.destroyPipeline(pipeline.pipeline, nullptr);
        init.disp.destroyPipelineLayout(pipeline.layout, nullptr);
        init.disp.destroyRenderPass(pipeline.renderPass, nullptr);

        swapchain.swapchain.destroy_image_views(swapchainImageViews);

        vkb::destroy_swapchain(swapchain.swapchain);
        vkb::destroy_device(init.device);
        vkb::destroy_instance(init.instance);
        glfwDestroyWindow(window);
        glfwTerminate();
    };

    void get_queues() {
        auto gq = init.device.get_queue(vkb::QueueType::graphics);
        if (!gq.has_value()) {
            std::cout << "failed to get graphics queue: " << gq.error().message() << "\n";
        }
        graphicsQueue = gq.value();

        auto pq = init.device.get_queue(vkb::QueueType::present);
        if (!pq.has_value()) {
            std::cout << "failed to get present queue: " << pq.error().message() << "\n";
        }
        presentQueue = pq.value();
    }

    void createSwapchain() {
        swapchain = Swapchain(&init, width, height);
    }

    void createGraphicsPipeline(const std::string& vertexPath, const std::string& fragmentPath) {
        pipeline = GraphicsPipeline(vertexPath, fragmentPath, &init, &swapchain, allocator, width, height);
    }

    void createFramebuffers() {
        swapchainImages = swapchain.swapchain.get_images().value();
        swapchainImageViews = swapchain.swapchain.get_image_views().value();

        framebuffers.resize(swapchainImageViews.size());

        for (size_t i = 0; i < swapchainImageViews.size(); i++) {
            VkImageView attachments[] = { swapchainImageViews[i] };

            VkFramebufferCreateInfo framebuffer_info = {};
            framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebuffer_info.renderPass = pipeline.renderPass;
            framebuffer_info.attachmentCount = 1;
            framebuffer_info.pAttachments = attachments;
            framebuffer_info.width = swapchain.swapchain.extent.width;
            framebuffer_info.height = swapchain.swapchain.extent.height;
            framebuffer_info.layers = 1;

            if (init.disp.createFramebuffer(&framebuffer_info, nullptr, &framebuffers[i]) != VK_SUCCESS) {
                throw std::exception("couldn't create framebuffers");
                return;
            }
        }
    }

    void createCommandPool() {
        VkCommandPoolCreateInfo pool_info = {};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.queueFamilyIndex = init.device.get_queue_index(vkb::QueueType::graphics).value();
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

        if (init.disp.createCommandPool(&pool_info, nullptr, &commandPool) != VK_SUCCESS) {
            std::cout << "failed to create command pool\n";
            return; // failed to create command pool
        }
    }

    void createCommandBuffers() {
        commandBuffers.resize(framebuffers.size());

        VkCommandBufferAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();

        if (init.disp.allocateCommandBuffers(&allocInfo, commandBuffers.data()) != VK_SUCCESS) {
            throw std::exception("couldn't allocate cmd bufs");
        }
        secondaryCmdBufs.resize(framebuffers.size());
        VkCommandBufferAllocateInfo allocInfo2 = {};
        allocInfo2.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo2.level = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
        allocInfo2.commandPool = commandPool;
        allocInfo2.commandBufferCount = (uint32_t)secondaryCmdBufs.size();

        if (init.disp.allocateCommandBuffers(&allocInfo2, secondaryCmdBufs.data()) != VK_SUCCESS) {
            throw std::exception("couldn't allocate secondary cmd bufs");
        }

        secondaryCmdBufsDepth.resize(framebuffers.size());
        VkCommandBufferAllocateInfo allocInfo3 = {};
        allocInfo3.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo3.level = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
        allocInfo3.commandPool = commandPool;
        allocInfo3.commandBufferCount = (uint32_t)secondaryCmdBufsDepth.size();
        if (init.disp.allocateCommandBuffers(&allocInfo3, secondaryCmdBufsDepth.data()) != VK_SUCCESS) {
            throw std::exception("couldn't allocate secondary cmd bufs");
        }

        recordPrimaryCmds();
    }

    void recordPrimaryCmds() {
        for (size_t i = 0; i < commandBuffers.size(); i++) {
            VkCommandBufferBeginInfo begin_info = {};
            begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

            if (init.disp.beginCommandBuffer(commandBuffers[i], &begin_info) != VK_SUCCESS) {
                throw std::exception("can't begin command buffer recording");
            }

            // viewport and scissor
            VkViewport viewport = {};
            viewport.x = 0.0f;
            viewport.y = 0.0f;
            viewport.width = (float)swapchain.swapchain.extent.width;
            viewport.height = (float)swapchain.swapchain.extent.height;
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;

            VkRect2D scissor = {};
            scissor.offset = { 0, 0 };
            scissor.extent = swapchain.swapchain.extent;

            init.disp.cmdSetViewport(commandBuffers[i], 0, 1, &viewport);
            init.disp.cmdSetScissor(commandBuffers[i], 0, 1, &scissor);

            // bind the depth pipeline first
            VkClearValue clearDepth{ { { 0.0f, 0 } } };
            pipeline.pipelines[0]->render(commandBuffers[i], secondaryCmdBufsDepth[i], clearDepth, i);

            // now bind the main pipeline
            VkRenderPassBeginInfo info = {};
            info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            info.framebuffer = framebuffers[i];
            info.clearValueCount = 1;
            VkClearValue clearColor{ { { 0.0f, 0.0f, 0.0f, 1.0f } } };
            info.pClearValues = &clearColor;
            info.renderArea.offset = { 0, 0 };
            info.renderArea.extent = swapchain.swapchain.extent;
            info.renderPass = pipeline.renderPass;

            pipeline.render(commandBuffers[i], secondaryCmdBufs[i], info);

            if (init.disp.endCommandBuffer(commandBuffers[i]) != VK_SUCCESS) {
                throw std::exception("couldn't end cmd buf");
            }
        }
    }

    void createSyncObjects() {
        availiables.resize(MAX_FRAMES_IN_FLIGHT);
        finishes.resize(MAX_FRAMES_IN_FLIGHT);
        inFlights.resize(MAX_FRAMES_IN_FLIGHT);
        imagesInFlights.resize(swapchain.swapchain.image_count, VK_NULL_HANDLE);

        VkSemaphoreCreateInfo semaphore_info = {};
        semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fence_info = {};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            if (init.disp.createSemaphore(&semaphore_info, nullptr, &availiables[i]) != VK_SUCCESS ||
                init.disp.createSemaphore(&semaphore_info, nullptr, &finishes[i]) != VK_SUCCESS ||
                init.disp.createFence(&fence_info, nullptr, &inFlights[i]) != VK_SUCCESS) {
                throw std::exception("failed to create sync objects");
            }
        }
    }

    void drawFrame() {
        init.disp.waitForFences(1, &inFlights[currentFrame], VK_TRUE, UINT64_MAX);

        uint32_t image_index = 0;
        VkResult result = init.disp.acquireNextImageKHR(
            swapchain.swapchain, UINT64_MAX, availiables[currentFrame], VK_NULL_HANDLE, &image_index);

        //if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        //    return recreate_swapchain(init, data);
        //}
        if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
            throw std::exception("failed to aquire next swapchain image");
        }

        if (imagesInFlights[image_index] != VK_NULL_HANDLE) {
            init.disp.waitForFences(1, &imagesInFlights[image_index], VK_TRUE, UINT64_MAX);
        }
        imagesInFlights[image_index] = inFlights[currentFrame];

        VkSubmitInfo submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        VkSemaphore wait_semaphores[] = { availiables[currentFrame] };
        VkPipelineStageFlags wait_stages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = wait_semaphores;
        submitInfo.pWaitDstStageMask = wait_stages;

        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[image_index];

        VkSemaphore signal_semaphores[] = { finishes[currentFrame] };
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signal_semaphores;

        init.disp.resetFences(1, &inFlights[currentFrame]);

        // Ensure secondary command buffers are reset before re-recording  
        init.disp.resetCommandBuffer(secondaryCmdBufsDepth[image_index], 0);
        init.disp.resetCommandBuffer(secondaryCmdBufs[image_index], 0);

        // Update secondary command buffers every frame for depth render pass  
        VkCommandBufferInheritanceInfo inheritanceInfoDepth = {};
        inheritanceInfoDepth.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO;
        inheritanceInfoDepth.renderPass = pipeline.pipelines[0]->renderPass;
        inheritanceInfoDepth.framebuffer = pipeline.pipelines[0]->framebuffer.framebuffers[image_index];
        inheritanceInfoDepth.pNext = nullptr;

        VkCommandBufferBeginInfo beginInfoDepth = {};
        beginInfoDepth.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfoDepth.flags = VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT;
        beginInfoDepth.pInheritanceInfo = &inheritanceInfoDepth;

        // Issue draw calls  
        init.disp.beginCommandBuffer(secondaryCmdBufsDepth[image_index], &beginInfoDepth);
        scene.renderSceneDepth(secondaryCmdBufsDepth[image_index], width, height);
        init.disp.endCommandBuffer(secondaryCmdBufsDepth[image_index]);

        // Update secondary command buffers every frame for main render pass  
        VkCommandBufferInheritanceInfo inheritanceInfo = {};
        inheritanceInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO;
        inheritanceInfo.renderPass = pipeline.renderPass;
        inheritanceInfo.framebuffer = framebuffers[image_index];

        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT;
        beginInfo.pInheritanceInfo = &inheritanceInfo;

        // Issue draw calls  
        init.disp.beginCommandBuffer(secondaryCmdBufs[image_index], &beginInfo);
        scene.renderScene(secondaryCmdBufs[image_index], width, height, image_index);
        init.disp.endCommandBuffer(secondaryCmdBufs[image_index]);

        if (init.disp.queueSubmit(graphicsQueue, 1, &submitInfo, inFlights[currentFrame]) != VK_SUCCESS) {
            throw std::exception("failed to submit frames to queue");
        }

        VkPresentInfoKHR present_info = {};
        present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

        present_info.waitSemaphoreCount = 1;
        present_info.pWaitSemaphores = signal_semaphores;

        VkSwapchainKHR swapChains[] = { swapchain.swapchain };
        present_info.swapchainCount = 1;
        present_info.pSwapchains = swapChains;

        present_info.pImageIndices = &image_index;

        result = init.disp.queuePresentKHR(presentQueue, &present_info);
        //if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
        //    return recreate_swapchain(init, data);
        //}
        if (result != VK_SUCCESS) {
            throw std::exception("failed to present to surface");
        }

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    // Store the previous state of the mouse buttons
    std::unordered_map<int, bool> mouseButtonStates;

    bool isMouseButtonPressed(GLFWwindow* window, int button) {
        int currentState = glfwGetMouseButton(window, button);

        // Check if the button was not pressed previously and is now pressed
        if (currentState == GLFW_PRESS && !mouseButtonStates[button]) {
            mouseButtonStates[button] = true;
            return true;
        }
        // If the button is released, remove it from the map
        else if (currentState == GLFW_RELEASE) {
            mouseButtonStates.erase(button);
        }

        return false;
    }

    std::unordered_map<int, bool> releaseMouseButtonStates;

    bool isMouseButtonReleased(GLFWwindow* window, int button) {
        int currentState = glfwGetMouseButton(window, button);

        // Check if the button was pressed previously and is now released
        if (currentState == GLFW_RELEASE && releaseMouseButtonStates[button]) {
            releaseMouseButtonStates[button] = false; // Update the state to released
            return true;
        }
        // If the button is pressed, update the map
        else if (currentState == GLFW_PRESS) {
            releaseMouseButtonStates[button] = true;
        }

        return false;
    }

    bool isMouseButtonPressedDown(GLFWwindow* window, int button) {
        if (glfwGetMouseButton(window, button)) return true;
        else return false;
    }

    glm::vec2 getCursorPosition(GLFWwindow* window) {
        double xPos;
        double yPos;
        glfwGetCursorPos(window, &xPos, &yPos);
        return glm::vec2(xPos, yPos);
    }

    std::unordered_map<int, bool> keyStates;

    bool isKeyPressed(GLFWwindow* window, int key) {
        int currentState = glfwGetKey(window, key);

        // Check if key was not pressed previously and is now pressed
        if (currentState == GLFW_PRESS && !keyStates[key]) {
            keyStates[key] = true;
            return true;
        }
        // Update the key state
        else if (currentState == GLFW_RELEASE) {
            keyStates.erase(key);
        }

        return false;
    }

    bool isKeyPressedDown(GLFWwindow* window, int key) {
        if (glfwGetKey(window, key) == GLFW_PRESS) {
            return true;
        }
        else return false;
    }

    static float xoffset;
    static float yoffset;
    static bool cursorEnabled;

    static void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
        static float lastX = 400, lastY = 300;
        static bool firstMouse = true;
        static float movementInterval = 0.005f; // Set interval duration in seconds (50 ms)
        static float lastMovementTime = 0.0f;  // Store the last time movement was triggered

        if (firstMouse) {
            lastX = xpos;
            lastY = ypos;
            firstMouse = false;
        }

        xoffset = xpos - lastX;
        yoffset = lastY - ypos;
        lastX = xpos;
        lastY = ypos;
        if (!cursorEnabled)
            scene.camera.ProcessMouseMovement(xoffset, -yoffset, true);
    }

    void toggleCursor(GLFWwindow* window, bool enableCursor) {
        if (enableCursor) {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);  // Show and unlock the cursor
            cursorEnabled = true;
        }
        else {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);  // Hide and lock the cursor
            cursorEnabled = false;
        }
    }

    float deltaTime = 0;

    void processInput(Camera& camera) {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);

        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            camera.ProcessKeyboard(FORWARD, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            camera.ProcessKeyboard(BACKWARD, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            camera.ProcessKeyboard(LEFT, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            camera.ProcessKeyboard(RIGHT, deltaTime);
    }

    void run() {
        auto lastTime = std::chrono::high_resolution_clock::now();

        while (!glfwWindowShouldClose(window)) {
            auto currentTime = std::chrono::high_resolution_clock::now();
            deltaTime = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - lastTime).count();
            lastTime = currentTime;

            if (isMouseButtonPressedDown(window, GLFW_MOUSE_BUTTON_2)) {
                toggleCursor(window, false);
            }
            else toggleCursor(window, true);

            processInput(scene.camera);

            drawFrame();

            //std::cout << glm::to_string(scene.camera.Position) << "\n";

            glfwPollEvents();

        }
        init.disp.deviceWaitIdle();
    }
};

bool Engine::cursorEnabled = true;
float Engine::xoffset = 0.0f;
float Engine::yoffset = 0.0f;
Scene Engine::scene{};
