require "Logger"
require "Linear"
require "ReLU"

local linear = Linear(192, 10)
local relu = ReLU()

linear.W = torch.load("../samples/W_sample_1.bin")[1]
linear.B = torch.load("../samples/B_sample_1.bin")[1]

local input = torch.load("../samples/input_sample_1.bin")
local gradOutput = torch.load("../samples/gradOutput_sample_1.bin")

local output = torch.load("../samples/output_sample_1.bin")
local gradW = torch.load("../samples/gradW_sample_1.bin")[1]
local gradB = torch.load("../samples/gradB_sample_1.bin")[1]
local target = torch.load("../samples/target_sample_1.bin")

local my_output = linear:forward(input:resize(5, 192))
local my_gradInput = linear:backward(input, gradOutput)
local my_gradW = linear.gradW
local my_gradB = linear.gradB

--logger:debug(output - my_output)
--logger:debug(gradW - my_gradW)
--logger:debug(gradB - my_gradB)
