require "src.Logger"
require "src.RNN"
require "src.Linear"
require "src.BatchNormalization"
require "src.Criterion"

function loadMappings(mappingFile)
	local mappings = {}
	local count = 0
	local file = io.open(mappingFile)
	if file then
		for line in file:lines() do
			local value, key = unpack(line:split("\t"))
			mappings[tonumber(key)] = tonumber(value)
			count = count + 1
		end
	end
	return mappings, count
end

function oneHotEncodeAux(vars, mappings, count)
	local oneHots = torch.zeros(count, #vars)
	for i = 1, #vars do
		oneHots[mappings[tonumber(vars[i])]][i] = 1
	end
	return oneHots
end

function oneHotEncode(filename, mappings, count)
	local input = {}
	local file = io.open(filename)
	if file then
		for line in file:lines() do
			table.insert(input, oneHotEncodeAux(line:split(" "), mappings, count))
		end
	end
	return input
end

local inputMappings, countInput = loadMappings("dataset/input_mappings")
local outputMappings, countOutput = loadMappings("dataset/output_mappings")
local inputs = oneHotEncode("dataset/train/train_data.txt", inputMappings, countInput)
local targets = oneHotEncode("dataset/train/train_labels.txt", outputMappings, countOutput)

local trainRatio = 0.8
local validationRatio = 1 - trainRatio

local indexPerms = torch.randperm(#inputs):long()
local trainIndexes = indexPerms:narrow(1, 1, indexPerms:size(1) * trainRatio)
local validationIndexes = indexPerms:narrow(1, trainIndexes:size(1) + 1, indexPerms:size(1) - trainIndexes:size(1))

local trainInputs = {unpack(inputs, 1, trainRatio * #inputs)}
local trainTargets = {unpack(targets, 1, trainRatio * #targets)}
local validationInputs = {unpack(inputs, trainRatio * #input + 1, #inputs)}
local validationTargets = {unpack(targets, trainRatio * #input + 1, #targets)}

rnn = RNN(countInput, 32)
linear = Linear(32, countOutput)
batchnorm = BatchNormalization()
criterion = Criterion()

local learningRate = 1e-6
local epochs = 5000

local gen = torch.Generator()
torch.manualSeed(gen, 0)
for epoch = 1, epochs do
	local i = torch.random(gen, 1, #trainInputs)
	rnn.clock = 0
	rnn.timestamp = 0
	for j = 1, trainInputs[i]:size(2) do
		rnn:forward(trainInputs[i]:select(2, j))
	end
	batchnorm:forward(rnn.output)
	linear:forward(batchnorm.output)
	criterion:forward(linear.output, trainTargets[i])
	if epoch % 25 == 0 then
		logger:debug(epoch .. "\t" .. rnn.clock .. "\t" .. criterion.output)
	end
	criterion:backward(linear.output, trainTargets[i])
	criterion:updateParameters(learningRate)
	linear:backward(batchnorm.output, criterion.gradInput)
	linear:updateParameters(learningRate)
	batchnorm:backward(rnn.output, linear.gradInput)
	batchnorm:updateParameters(learningRate)
	local back = linear.gradInput
	for j = trainInputs[i]:size(2), 1, -1 do
		rnn:backward(trainInputs[i]:select(2, j), back)
		back = rnn.gradInput
	end
	rnn:updateParameters(learningRate)
	if epoch % 100 == 0 then
		local errorSums = 0
		for k = 1, #validationInputs do
			for l = 1, validationInputs[k]:size(2) do
				rnn:forward(validationInputs[k]:select(2, l))
			end
			batchnorm:forward(rnn.output)
			linear:forward(batchnorm.output)
			if linear.output[1][1] > linear.output[2][1] then
				linear.output[1][1] = 1
				linear.output[2][1] = 0
			else
				linear.output[1][1] = 0
				linear.output[2][1] = 1
			end
			errorSums = errorSums + torch.abs(linear.output - validationTargets[k])
		end
		logger:info("Error: " .. torch.sum(errorSums / #validationInputs))
	end
end
