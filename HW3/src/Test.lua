require "Logger"
require "Linear"
require "ReLU"
require "Criterion"
require "Model"
require "GradientDescent"
require "BatchNormalization"
require "SpatialConvolution"

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

function predict(model)
	if model == nil then
		logger:error("No model given")
		error()
	end
	local model = torch.load(model)
	local testData = torch.load("../dataset/Test/test.bin"):double()
	local output = model:forward(testData:resize(testData:size(1), testData:size(2) * testData:size(3)))
	local value, index = torch.max(output, 2)
	local indexfile = io.open(os.date("prediction_index_%Y-%m-%d-%X.csv"), "w")
	indexfile:write("id,label\n")
	for i = 1, value:size(1) do
		indexfile:write(i - 1 .. "," .. index[i][1] - 1 .. "\n")
	end
	io.close(indexfile)
end

function train()
	local input = torch.load("../dataset/Train/data.bin"):double()
	local target = torch.load("../dataset/Train/labels.bin"):double()
	local target_onehot = torch.zeros(target:size(1), 6)
	for i = 1, target:size(1) do
		target_onehot[i][target[i] + 1] = 1
	end

	input = input / 255
	local mean = torch.mean(input, 1)
	-- local stddev = torch.std(input, 1)

	for i = 1, input:size(1) do
		input[i] = input[i] - mean
	end

	local mlp;
	local model = ""
	if model == "" then
		mlp = Model()
		mlp:addLayer(Linear(11664, 10))
		mlp:addLayer(BatchNormalization())
		-- mlp:addLayer(ReLU())
		mlp:addLayer(Linear(10, 6))
	else
		mlp = torch.load(model)
		logger:info("Model loaded from file " .. model)
	end

	local learningRate = 1e-5
	local maxIterations = 10
	local batchSize = 100
	local criterion = Criterion()
	logger:info(mlp)
	logger:info("Learning Rate = " .. learningRate)
	logger:info("Iterations = " .. maxIterations)
	logger:info("Batch Size = " .. batchSize)
	-- local x = mlp:forward(input:narrow(1, 1, 2))
	-- local loss = criterion:forward(x, target:narrow(1, 1, 2))
	-- local y = criterion:backward(x, target:narrow(1, 1, 2))
	-- mlp:backward(input:narrow(1, 1, 2), y)
	-- print(loss)

	local trainer = GradientDescent(mlp, criterion, learningRate, maxIterations)
	trainer:train(input, target_onehot, batchSize)

	local saveModel = {}
	table.insert(saveModel,mlp);
	table.insert(saveModel,criterion);
	table.insert(saveModel,learningRate);
	table.insert(saveModel,maxIterations);
  
	local filename = os.date("MLP_%Y-%m-%d-%X.bin")
	logger:info("Saving the model as ".. filename)
	torch.save(filename, mlp)
	predict(filename)
end

-- model1()
-- model2()
train()
-- predict("MLP_2018-02-21-10:38:25.bin")
