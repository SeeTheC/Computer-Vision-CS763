require "src.Logger"
require "src.Helper"
require "src.Model"
require "src.RNN"
require "src.BatchNormalization"
require "src.Linear"
require "src.Criterion"
require "src.SGD"

-- Load and processing data
local inputMappings, countInput = loadMappings("dataset/input_mappings")
local outputMappings, countOutput = loadMappings("dataset/output_mappings")
local inputs = oneHotEncode("dataset/train/train_data.txt", inputMappings, countInput)
local targets = oneHotEncode("dataset/train/train_labels.txt", outputMappings, countOutput)

local trainRatio = 0.8
local validationRatio = 1 - trainRatio

local trainInputs = {unpack(inputs, 1, trainRatio * #inputs)}
local trainTargets = {unpack(targets, 1, trainRatio * #targets)}
local validationInputs = {unpack(inputs, trainRatio * #inputs + 1, #inputs)}
local validationTargets = {unpack(targets, trainRatio * #targets + 1, #targets)}

-- Defining the model
local rnnHiddenDimension = 512
local inputDimension = 153
local outputDimension = 2
local modelLoadPath = "rnn_2018-03-16-17:59:30.bin"

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
	model:add(Linear(inputDimension, outputDimension))
end

local criterion = Criterion()

local epochs = 2000
local learningRate = 1e-2
local accuracyAfterEpochs = 100
local saveModelAfterEpochs = 1000
local saveModelPathPrefix = os.date("rnn_%Y-%m-%d-%X" .. rnnHiddenDimension)

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
