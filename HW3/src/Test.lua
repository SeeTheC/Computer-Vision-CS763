require "Logger"
require "Linear"
require "ReLU"
require "Criterion"

local function equal(matrix1, matrix2)
	if (math.abs(torch.sum(matrix1 - matrix2)) < 1e-6) then
		return "Equal"
	else
		return "Not Equal"
	end
end

-- Sample 1
logger:info("Running Sample 1")
local linear1 = Linear(192, 10)
local criterion = Criterion()

linear1.W = torch.load("../samples/W_sample_1.bin")[1]
linear1.B = torch.load("../samples/B_sample_1.bin")[1]

local input = torch.load("../samples/input_sample_1.bin")
local gradOutput = torch.load("../samples/gradOutput_sample_1.bin")

local input_criterion = torch.load("../samples/input_criterion_sample_1.bin")
local gradCriterionInput = torch.load("../samples/gradCriterionInput_sample_1.bin")

local output = torch.load("../samples/output_sample_1.bin")
local gradW1 = torch.load("../samples/gradW_sample_1.bin")[1]
local gradB1 = torch.load("../samples/gradB_sample_1.bin")[1]
local target = torch.load("../samples/target_sample_1.bin")

local linear1_output = linear1:forward(input:resize(5, 192))

local linear1_gradInput = linear1:backward(input:resize(5, 192), gradOutput)

local linear1_gradW = linear1.gradW
local linear1_gradB = linear1.gradB

local criterion_output = criterion:forward(input_criterion, target)
local criterion_gradInput = criterion:backward(input_criterion, target)

logger:debug("Output  : " .. equal(output, linear1_output))
logger:debug("GradW1  : " .. equal(gradW1, linear1_gradW))
logger:debug("GradB1  : " .. equal(gradB1, linear1_gradB))
logger:debug("Criterion GradInput : " .. equal(gradCriterionInput, criterion_gradInput))


-- Sample 2
logger:info("Running Sample 2")
local linear1 = Linear(192, 10)
local relu1 = ReLU()
local linear2 = Linear(10, 3)
local criterion = Criterion()

linear1.W = torch.load("../samples/W_sample_2.bin")[1]
linear1.B = torch.load("../samples/B_sample_2.bin")[1]

linear2.W = torch.load("../samples/W_sample_2.bin")[2]
linear2.B = torch.load("../samples/B_sample_2.bin")[2]

local input = torch.load("../samples/input_sample_2.bin")
local gradOutput = torch.load("../samples/gradOutput_sample_2.bin")

local input_criterion = torch.load("../samples/input_criterion_sample_2.bin")
local gradCriterionInput = torch.load("../samples/gradCriterionInput_sample_2.bin")

local output = torch.load("../samples/output_sample_2.bin")
local gradW1 = torch.load("../samples/gradW_sample_2.bin")[1]
local gradB1 = torch.load("../samples/gradB_sample_2.bin")[1]
local gradW2 = torch.load("../samples/gradW_sample_2.bin")[2]
local gradB2 = torch.load("../samples/gradB_sample_2.bin")[2]
local target = torch.load("../samples/target_sample_2.bin")

local linear1_output = linear1:forward(input:resize(5, 192))
local relu1_output = relu1:forward(linear1_output)
local linear2_output = linear2:forward(relu1_output)

local linear2_gradInput = linear2:backward(relu1_output, gradOutput)
local relu1_gradInput = relu1:backward(linear1_output, linear2_gradInput)
local linear1_gradInput = linear1:backward(input:resize(5, 192), relu1_gradInput)

local linear1_gradW = linear1.gradW
local linear1_gradB = linear1.gradB
local linear2_gradW = linear2.gradW
local linear2_gradB = linear2.gradB

local criterion_output = criterion:forward(input_criterion, target)
local criterion_gradInput = criterion:backward(input_criterion, target)

logger:debug("Output  : " .. equal(output, linear2_output))
logger:debug("GradW1  : " .. equal(gradW1, linear1_gradW))
logger:debug("GradB1  : " .. equal(gradB1, linear1_gradB))
logger:debug("GradW2  : " .. equal(gradW2, linear2_gradW))
logger:debug("GradB2  : " .. equal(gradB2, linear2_gradB))
logger:debug("Criterion GradInput : " .. equal(gradCriterionInput, criterion_gradInput))
