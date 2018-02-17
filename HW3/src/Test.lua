require "Logger"
require "Linear"
require "ReLU"
require "Criterion"
require "Model"

local function equal(matrix1, matrix2)
	if (math.abs(torch.sum(matrix1 - matrix2)) < 1e-6) then
		return "Equal"
	else
		return "Not Equal"
	end
end

-- Sample 1
function model1()
	logger:info("Running Sample 1")
	local nn = Model()
	local criterion = Criterion()
	nn:addLayer(Linear(192, 10))

	nn.Layers[1].W = torch.load("../samples/W_sample_1.bin")[1]
	nn.Layers[1].B = torch.load("../samples/B_sample_1.bin")[1]

	local input = torch.load("../samples/input_sample_1.bin")
	local gradOutput = torch.load("../samples/gradOutput_sample_1.bin")

	local input_criterion = torch.load("../samples/input_criterion_sample_1.bin")
	local gradCriterionInput = torch.load("../samples/gradCriterionInput_sample_1.bin")

	local output = torch.load("../samples/output_sample_1.bin")
	local gradW1 = torch.load("../samples/gradW_sample_1.bin")[1]
	local gradB1 = torch.load("../samples/gradB_sample_1.bin")[1]
	local target = torch.load("../samples/target_sample_1.bin")

	local nn_output = nn:forward(input:resize(5, 192))
	nn:backward(input:resize(5, 192), gradOutput)

	local linear1_gradW = nn.Layers[1].gradW
	local linear1_gradB = nn.Layers[1].gradB

	local criterion_output = criterion:forward(input_criterion, target)
	local criterion_gradInput = criterion:backward(input_criterion, target)

	logger:debug("Output  : " .. equal(output, nn_output))
	logger:debug("GradW1  : " .. equal(gradW1, linear1_gradW))
	logger:debug("GradB1  : " .. equal(gradB1, linear1_gradB))
	logger:debug("Criterion GradInput : " .. equal(gradCriterionInput, criterion_gradInput))
end

-- Sample 2
function model2()
	logger:info("Running Sample 2")
	local nn = Model()
	local criterion = Criterion()
	nn:addLayer(Linear(192, 10))
	nn:addLayer(ReLU())
	nn:addLayer(Linear(10, 3))

	nn.Layers[1].W = torch.load("../samples/W_sample_2.bin")[1]
	nn.Layers[1].B = torch.load("../samples/B_sample_2.bin")[1]

	nn.Layers[3].W = torch.load("../samples/W_sample_2.bin")[2]
	nn.Layers[3].B = torch.load("../samples/B_sample_2.bin")[2]

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

	local nn_output = nn:forward(input:resize(5, 192))
	nn:backward(input:resize(5, 192), gradOutput)

	local linear1_gradW = nn.Layers[1].gradW
	local linear1_gradB = nn.Layers[1].gradB
	local linear2_gradW = nn.Layers[3].gradW
	local linear2_gradB = nn.Layers[3].gradB

	local criterion_output = criterion:forward(input_criterion, target)
	local criterion_gradInput = criterion:backward(input_criterion, target)

	logger:debug("Output  : " .. equal(output, nn_output))
	logger:debug("GradW1  : " .. equal(gradW1, linear1_gradW))
	logger:debug("GradB1  : " .. equal(gradB1, linear1_gradB))
	logger:debug("GradW2  : " .. equal(gradW2, linear2_gradW))
	logger:debug("GradB2  : " .. equal(gradB2, linear2_gradB))
	logger:debug("Criterion GradInput : " .. equal(gradCriterionInput, criterion_gradInput))
end

model1()
model2()
