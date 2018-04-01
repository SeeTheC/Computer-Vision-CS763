require "xlua"
require "Logger"
require "Linear"
require "ReLU"
require "Criterion"
require "Model"
require "GradientDescent"
require "BatchNormalization"
require "SpatialConvolution"

cmd = torch.CmdLine()

local cmd = torch.CmdLine()
if not opt then
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Options:')
   cmd:option("-modelName" ,"" ,"<model-name>")
   cmd:option("-data" ,"data.bin" ,"/path/to/data.bin")
   cmd:option("-target" ,"labels.bin" ,"/path/to/target/labels.bin")
   cmd:text()
   opt = cmd:parse(arg or {})
end


function train()
	local input = torch.load(opt["data"]):double()
	local target = torch.load(opt["target"]):double()
	os.execute("rm -rf " .. opt["modelName"])
	os.execute("mkdir " .. opt["modelName"])

	target = target + 1

	input = input / 255
	local mean = torch.mean(input, 1)
	local stddev = torch.std(input, 1)

	for i = 1, input:size(1) do
		input[i] = input[i] - mean
	end

	local mlp

	local model = ""
	if model == "" then
		mlp = Model()
		mlp:addLayer(Linear(11664, 10))
		mlp:addLayer(BatchNormalization())
		mlp:addLayer(ReLU())
		mlp:addLayer(Linear(10, 6))
	else
		mlp = torch.load(model)
		logger:info("Model loaded from file " .. model)
	end

	local learningRate = 1.5
	local maxIterations = 300
	local batchSize = 250

	local criterion = Criterion()

	logger:info(mlp)
	logger:info("Learning Rate = " .. learningRate)
	logger:info("Iterations = " .. maxIterations)
	logger:info("Batch Size = " .. batchSize)

	local trainer = GradientDescent(mlp, criterion, learningRate, maxIterations)
	trainer:train(input, target, batchSize)

	local saveModel = {}
	table.insert(saveModel,mlp);
	table.insert(saveModel,criterion);
	table.insert(saveModel,learningRate);
	table.insert(saveModel,maxIterations);

	local filename = os.date("MLP_%Y-%m-%d-%X.bin")
	logger:info("Saving the model as ".. filename)
	torch.save(opt["modelName"] .. "/model.bin", saveModel)

end


function train()
	local input = torch.load(opt["data"]):double()
	local target = torch.load(opt["target"]):double()
	os.execute("rm -rf " .. opt["modelName"])
	os.execute("mkdir " .. opt["modelName"])
	local target_onehot = torch.zeros(target:size(1), 6)
	for i = 1, target:size(1) do
		target_onehot[i][target[i] + 1] = 1
	end


	input = input / 255
	local mean = torch.mean(input, 1)
	local stddev = torch.std(input, 1)

	for i = 1, input:size(1) do
		input[i] = input[i] - mean
	end

	local mlp

	local model = ""
	if model == "" then
		mlp = Model()
		mlp:addLayer(Linear(11664, 10))
		mlp:addLayer(BatchNormalization())
		mlp:addLayer(ReLU())
		mlp:addLayer(Linear(10, 6))
	else
		mlp = torch.load(model)
		logger:info("Model loaded from file " .. model)
	end

	local learningRate = 1
	local maxIterations = 1000
	local batchSize = 100

	local criterion = Criterion()

	logger:info(mlp)
	logger:info("Learning Rate = " .. learningRate)
	logger:info("Iterations = " .. maxIterations)
	logger:info("Batch Size = " .. batchSize)


	local trainer = GradientDescent(mlp, criterion, learningRate, maxIterations)
	trainer:train(input, target_onehot, batchSize)

	local saveModel = {}
	table.insert(saveModel,mlp);
	table.insert(saveModel,criterion);
	table.insert(saveModel,learningRate);
	table.insert(saveModel,maxIterations);

	torch.save(opt["modelName"] .. "/model.bin", saveModel)
end

train()
