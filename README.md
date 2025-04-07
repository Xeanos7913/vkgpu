# vkgpu
This library makes the process of using vulkan compute shaders easy and seamless. I created this library as a project to see if I could mimic the power of Kompute, the popular Vulkan-based gpgpu library, and this is what I got. It has a single-gpu task struct, called "gpuTask<T>" where T is both its input and output types. Currently only supports floats, ints, doubles. gpuTask can currently have only 1 output.

## Compute Sequence
The next one, the most interesting one, is the ComputeSequence struct. It intakes SequentialGpuTask structs and executes them one by one sequentially, using VkSemaphores for given iterations. It uses efficient memory-pooling optimizations on the device-local memory, and the outputs of one task can be the inputs of the next task. Potentially useful for machine learning purposes where there is a need to do multiple matrix multiplies sequentially.

## Example for single gpuTask:

```cpp
#include "vkgpu.hpp"
#include <iostream>

int main(void) {

    // inputs for the single GPU task
    auto vec1 = std::vector<int>{ 1, 1, 1 };
    auto vec2 = std::vector<int>{ 5, 5, 5 };

    auto input = std::vector<std::vector<int>>{ vec1, vec2};

    auto singleTaskCode = readShaderBytecode("shader.spv");

    Init init;
    device_initialization(init);

    // single GPU task
    gpuTask<int> task = gpuTask<int>(singleTaskCode, init, 6);
    
    task.initiateCompute();
    task.load_inputs(input);
    auto result = task.compute();
    
    std::cout << "\Single Result: \n\t";
    for (auto num : result) { std::cout << num << " "; }
    return 0;
}
```

The shader used in this example is this:
```glsl
#version 450

// Define input and output buffers
layout(set = 0, binding = 0) readonly buffer InputA {
    int a[3];
};

layout(set = 0, binding = 1) readonly buffer InputB {
    int b[3];
};

layout(set = 0, binding = 2) writeonly buffer Output {
    int result[3];
};

void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index < 3) {  // Ensure within bounds
        result[index] = a[index] * a[index] + b[index] * b[index];
    }
}
```

## Example for the ComputeSequence:

For ComputeSequence, things are slightly more complicated. For this, you need to create all the buffers you want on the host-side, using std::vector<T>. Then, you need to pack all these vectors inside another vector, and then pass that into the ComputeSequence struct after initializing its memory. For each task, you actually need to create two vectors of integers, the inputIndices and the outputIndices vectors. These vectors are then loaded into the SequentialGpuTask struct and they basically dictate that which task can access which buffers as inputs and which buffers as outputs.

Here is a simple example:

```cpp
#include "vkgpu.hpp"
#include <iostream>

int main(void){
    // inputs for the sequential GPU task
    std::vector<float> buffer1{2.0f, 2.0f, 2.0f};
    std::vector<float> buffer2{1.0f, 1.0f, 1.0f};
    std::vector<float> buffer3{ 1.0f, 1.0f, 1.0f };
    std::vector<float> buffer4{ 1.0f, 1.0f, 1.0f };

    // indices for task1
    std::vector<uint32_t> inputIndices{ 0 };
    std::vector<uint32_t> outputIndices{ 1 };

    // indices for task2
    std::vector<uint32_t> inputIndices2{ 1 };
    std::vector<uint32_t> outputIndices2{ 2 };

    // indices for task3
    std::vector<uint32_t> inputIndices3{ 2 };
    std::vector<uint32_t> outputIndices3{ 3 };
    
    // create input buffers
    std::vector<std::vector<float>> buffers{buffer1, buffer2, buffer3, buffer4};
    
    // sequential task shader codes
    auto code = readShaderBytecode("task1.spv");
    auto code1 = readShaderBytecode("task2.spv");
    auto code2 = readShaderBytecode("task3.spv");
    
    Init init;
    
    device_initialization(init);

    // sequential GPU task
    auto task1 = SequentialgpuTask<float>(code, init);
    auto task2 = SequentialgpuTask<float>(code1, init);
    auto task3 = SequentialgpuTask<float>(code2, init);
    
    task1.load_indices(inputIndices, outputIndices);
    task2.load_indices(inputIndices2, outputIndices2);
    task3.load_indices(inputIndices3, outputIndices3);

    // the 24 means 24 bytes of total buffers.
    auto sequence = ComputeSequence<float>(&init, 24);

    // initialize the memory
    sequence.initMemory(24);

    // add the tasks in the order you wish too execute them:
    sequence.addTask(task1);
    sequence.addTask(task2);
    sequence.addTask(task3);

    // add the buffers:
    sequence.addBuffers(buffers);

    // initialize all the vulkan backend stuff:
    sequence.initializeCompute();

    // do the tasks for 1 iteration over a workgroup size of 3, 1, 1:
    sequence.doTasks(1, 3, 1, 1);

    // simply retrieve the buffer you want to, after the computation:
    auto resultSeq = sequence.allBuffers[3];

    // print results:
    std::cout << "\nSequential Result: \n\t";
    for (auto num : resultSeq) { std::cout << num << " "; }

    return 0;
}
```
Here are the shaders used for the tasks:

Task1:
```glsl
#version 460

#define numInputs 1
#define numOutputs 1

layout(set = 0, binding = 0) readonly buffer Input {
    float a[];
};

layout(set = 0, binding = 1) writeonly buffer Output {
    float result[];
};

void main() {
    int index = int(gl_GlobalInvocationID.x);
    result[index] = a[index] * a[index];
}
```
Task2:
```glsl
#version 460

#define numInputs 1
#define numOutputs 1

layout(set = 0, binding = 1) readonly buffer Input {
    float a[];
};

layout(set = 0, binding = 2) writeonly buffer Output {
    float result[];
};

void main() {
    int index = int(gl_GlobalInvocationID.x);
    result[index] = a[index] * a[index];
}
```
Task3:
```glsl
#version 460

#define numInputs 1
#define numOutputs 1

layout(set = 0, binding = 2) readonly buffer Input {
    float a[];
};

layout(set = 0, binding = 3) writeonly buffer Output {
    float result[];
};

void main() {
    int index = int(gl_GlobalInvocationID.x);
    result[index] = a[index] * a[index];
}
```
Notice how the set number for each buffer is 0, but the binding number is the index of the buffer which you loaded in. This is crutial for it to work.

## Dependencies:
This library uses VkBootstrap (https://github.com/charles-lunarg/vk-bootstrap) and the vulkan sdk's includes.

## Future Plans:
1. Support Tensors and other commonly used datatypes.
2. Use Vulkan's powerful Buffer Device Address, or Buffer Reference for memory access in the shaders. This will eliminate the need for Descriptors and thereby potentially reduce CPU bottleneck.
3. Expand the built-in memory manager to do some more fancy stuff, like paging and perhaps even sparse buffers.
    
