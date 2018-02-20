require "Logger"
require "Linear"
require "ReLU"
require "Criterion"
require "Model"
require "GradientDescent"
require "BatchNormalization"

local function equal(matrix1, matrix2)
	if (torch.sum((matrix1 - matrix2)) < 1e-9) then
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

function rmse(m1, m2)
	return torch.sqrt(torch.sum(torch.pow(m1 - m2, 2)))
end

function train()
	local input = torch.load("../dataset/Train/data.bin"):double()
	local target = torch.load("../dataset/Train/labels.bin"):double()
	target = target + 1

	local mean = torch.mean(input, 1)
	local stddev = torch.std(input, 1)

	for i = 1, input:size(1) do
		input[i] = input[i] - mean
	end

	local mlp

	local newModel = true
	if newModel then
		mlp = Model()
		mlp:addLayer(Linear(11664, 50))
		mlp:addLayer(BatchNormalization())
		mlp:addLayer(ReLU())
		mlp:addLayer(Linear(50, 10))
		mlp:addLayer(BatchNormalization())
		mlp:addLayer(ReLU())
		mlp:addLayer(Linear(10, 6))
	else
		local modelFile = "MLP_2018-02-20-16:22:34.bin"
		mlp = torch.load(modelFile)
		logger:info("Model loaded from file " .. modelFile)
	end

	local learningRate = 1e-1
	local maxIterations = 8500
	local batchSize = 729

	local criterion = Criterion()

	logger:info(mlp)
	logger:info("Learning Rate = " .. learningRate)
	logger:info("Iterations = " .. maxIterations)
	logger:info("Batch Size = " .. batchSize)

	local trainer = GradientDescent(mlp, criterion, learningRate, maxIterations)
	trainer:train(input, target, batchSize)

	local filename = os.date("MLP_%Y-%m-%d-%X.bin")
	logger:info("Saving the model as ".. filename)
	torch.save(filename, mlp)
end


-- model1()
-- model2()
train()
