require "src.Logger"
require "src.Helper"
require "src.Model"
require "src.RNN"
require "src.BatchNormalization"
require "src.Linear"
require "src.Criterion"
require "src.SGD"

local cmd = torch.CmdLine()
cmd:text()
cmd:text("Options:")
cmd:option("-modelName", "", "/path/to/model")
cmd:option("-rnnSize", 103, "Size of Hidden Layer")
cmd:option("-data", "", "/path/to/input")
cmd:option("-target", "", "/path/to/target")
cmd:text()
opt = cmd:parse(arg or {})

local modelName = opt["modelName"]
local rnnHiddenDimension = tonumber(opt["rnnSize"])
local inputFileName = opt["data"]
local targetFileName = opt["target"]

if modelName == "" then
	modelName = os.date("rnn_%Y-%m-%d-%X_" .. rnnHiddenDimension .. "/")
else
	modelName = modelName .. "/"
end
if inputFileName == "" or targetFileName == "" then
	cmd:help()
	os.exit()
end

-- Load and processing data
local inputMappings, countInput = loadMappings("src/input_mappings")
local outputMappings, countOutput = loadMappings("src/output_mappings")
local inputs = oneHotEncode(inputFileName, inputMappings, countInput)
local targets = oneHotEncode(targetFileName, outputMappings, countOutput)

local trainRatio = 0.8
local validationRatio = 1 - trainRatio

local trainInputs = {unpack(inputs, 1, trainRatio * #inputs)}
local trainTargets = {unpack(targets, 1, trainRatio * #targets)}
local validationInputs = {unpack(inputs, trainRatio * #inputs + 1, #inputs)}
local validationTargets = {unpack(targets, trainRatio * #targets + 1, #targets)}

-- Defining the model
-- local rnnHiddenDimension = 512
local inputDimension = countInput
local outputDimension = countOutput
local modelLoadPath = nil
if paths.dir(modelName) == nil then
	paths.mkdir(modelName)
else
	modelLoadPath = modelName .. "model.bin"
end
logger:setLogFilePrefix(modelName)

local model
if modelLoadPath then
	logger:info("Load Model Path: " .. modelLoadPath)
	model = torch.load(modelLoadPath)
else
	model = Model()
	model.H = rnnHiddenDimension
	model.V = inputDimension
	model:add(RNN(inputDimension, rnnHiddenDimension))
	model:add(BatchNormalization())
	model:add(Linear(rnnHiddenDimension, outputDimension))
end

local criterion = Criterion()

local epochs = 5000
local learningRate = 1e-2
local accuracyAfterEpochs = 250
local saveModelAfterEpochs = 1000
local saveModelPathPrefix = modelName

logger:info(model)
logger:info("Epochs: " .. epochs)
logger:info("Learning Rate: " .. learningRate)
logger:info("Accuracy after epochs: " .. accuracyAfterEpochs)
logger:info("Save model after epochs: " .. saveModelAfterEpochs)
logger:info("Save model path prefix: " .. saveModelPathPrefix)

-- Stochastic Gradient Descent
local sgd = SGD(model,
                criterion,
                epochs,
                learningRate,
                trainInputs,
                trainTargets,
                validationInputs,
                validationTargets,
                accuracyAfterEpochs,
                saveModelAfterEpochs,
                saveModelPathPrefix
               )
sgd:train()
