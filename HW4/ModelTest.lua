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
local model = Model()
local rnn = RNN(153, 32)
local batchnorm = BatchNormalization()
local linear = Linear(32, 2)
model:add(rnn)
model:add(batchnorm)
model:add(linear)

local criterion = Criterion()

local epochs = 5e3
local learningRate = 1e-3
local accuracyAfterEpochs = 100

local sgd = SGD(model, criterion, epochs, learningRate, trainInputs, trainTargets, validationInputs, validationTargets, accuracyAfterEpochs)
sgd:train()
