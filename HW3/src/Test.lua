require "Logger"
require "Linear"
require "ReLU"
require "Criterion"
require "Model"
require "image"
require "nn"

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

function train()
	local nn = Model()
	nn:addLayer(Linear(11664, 6))
	--nn:addLayer(ReLU())
	--nn:addLayer(Linear(150, 30))
	--nn:addLayer(ReLU())
	--nn:addLayer(Linear(30, 6))
	local criterion = Criterion()

	logger:debug("Loading the dataset")
	local input = torch.load("../dataset/Train/data.bin"):double()
	local target = torch.load("../dataset/Train/labels.bin"):double()
	for i = 1, target:size(1) do
		target[i] = target[i] + 1
	end
	logger:debug("Dataset loaded")

	local n_images = input:size(1)
	local height = input:size(2)
	local width = input:size(3)

	logger:debug("Normalizing Data")
	local mean = torch.mean(input, 1)
	local stddev = torch.std(input, 1)

	for i = 1, n_images do
		input[i] = torch.cdiv((input[i] - mean), stddev)
	end
	logger:debug("Normalizing finished")

	logger:debug("Beginning Gradient Descent")
	for epoch = 1, 20 do
		logger:debug("Forward in epoch " .. epoch)
		local output = nn:forward(input:resize(n_images, height * width))
		local criterion_output = criterion:forward(output, target)
		if epoch then
			logger:info("Epochs: " .. epoch .. ", Loss: " .. criterion_output)
		end
		logger:debug("Forward end")
		logger:debug("Backwards in epoch " .. epoch)
		local criterion_gradInput = criterion:backward(output, target)
		nn:backward(input:resize(n_images, height * width), criterion_gradInput)
		logger:debug("Backwards end")
	end
	logger:debug("Gradient Descent converged")
end

function rmse(m1, m2)
	return torch.sqrt(torch.sum(torch.pow(m1 - m2, 2)))
end

function train_lib()
	local nn1 = Model()
	local nn2 = nn.Sequential()
	nn1:addLayer(Linear(11664, 150))
	nn1:addLayer(ReLU())
	nn1:addLayer(Linear(150, 30))
	nn1:addLayer(ReLU())
	nn1:addLayer(Linear(30, 6))

	local nn_linear1 = nn.Linear(11664, 150)
	local nn_linear2 = nn.Linear(150, 30)
	local nn_linear3 = nn.Linear(30, 6)
	nn2:add(nn_linear1)
	nn2:add(nn.ReLU())
	nn2:add(nn_linear2)
	nn2:add(nn.ReLU())
	nn2:add(nn_linear3)

	nn1.Layers[1].W = nn_linear1.weight
	nn1.Layers[1].B = nn_linear1.bias
	nn1.Layers[3].W = nn_linear2.weight
	nn1.Layers[3].B = nn_linear2.bias
	nn1.Layers[5].W = nn_linear3.weight
	nn1.Layers[5].B = nn_linear3.bias
	
	local criterion1 = Criterion()
	local criterion2 = nn.CrossEntropyCriterion()

	logger:debug("Loading the dataset")
	local input = torch.load("../dataset/Train/data.bin"):double()
	local target = torch.load("../dataset/Train/labels.bin"):double()
	for i = 1, target:size(1) do
		target[i] = target[i] + 1
	end
	logger:debug("Dataset loaded")

	local n_images = input:size(1)
	local height = input:size(2)
	local width = input:size(3)

	logger:debug("Normalizing Data")
	local mean = torch.mean(input, 1)
	local stddev = torch.std(input, 1)

	for i = 1, n_images do
		input[i] = torch.cdiv((input[i] - mean), stddev)
	end
	logger:debug("Normalizing finished")

	logger:debug("Beginning Gradient Descent")
	for epoch = 1, 20 do
		logger:debug("Forward in epoch " .. epoch)
		local output1 = nn1:forward(input:resize(n_images, height * width))
		local output2 = nn2:forward(input:resize(n_images, height * width))
		logger:debug("Output: " .. rmse(output1, output2))
		local criterion1_output = criterion1:forward(output1, target)
		local criterion2_output = criterion2:forward(output2, target)
		logger:debug("Criterion: " .. (criterion1_output - criterion2_output))
		if epoch then
			logger:info("Epochs: " .. epoch .. ", Loss: " .. criterion1_output)
			logger:info("Epochs: " .. epoch .. ", Loss: " .. criterion2_output)
		end
		logger:debug("Forward end")
		logger:debug("Backwards in epoch " .. epoch)
		local criterion1_gradInput = criterion1:backward(output1, target)
		local criterion2_gradInput = criterion2:backward(output2, target)
		logger:debug("Criterion GradInput: " .. rmse(criterion1_gradInput, criterion2_gradInput))

		nn1:backward(input:resize(n_images, height * width), criterion1_gradInput)
		nn2:backward(input:resize(n_images, height * width), criterion2_gradInput)
		logger:debug("Backwards end")
	end
	logger:debug("Gradient Descent converged")
end


--model1()
--model2()
--train()
train_lib()
