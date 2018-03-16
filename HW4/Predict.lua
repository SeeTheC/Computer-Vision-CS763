require "src.Logger"
require "src.Helper"
require "src.Model"
require "src.RNN"
require "src.BatchNormalization"
require "src.Linear"

-- Load and processing data
local inputMappings, countInput = loadMappings("dataset/input_mappings")
local outputMappings, countOutput = loadMappings("dataset/output_mappings")
local inputs = oneHotEncode("dataset/test/test_data.txt", inputMappings, countInput)

local modelLoadPath = "rnn_2018-03-16-17:59:30.bin"

if modelLoadPath then
	logger:debug("Load Model Path: " .. modelLoadPath)
	model = torch.load(modelLoadPath)
	local predictedOutputsFileName = modelLoadPath .. "_predictions.csv"
	predictedOutputsFile = io.open(predictedOutputsFileName, "w")
	predictedOutputsFile:write("id,label\n")
	for i = 1, #inputs do
		model:resetGradsAndOutputs()
		local output = model:predict(inputs[i])
		if output[1][1] > output[2][1] then
			predictedOutputsFile:write((i - 1) .. "," .. 0 .. "\n")
		else
			predictedOutputsFile:write((i - 1) .. "," .. 1 .. "\n")
		end
	end
	predictedOutputsFile.close()
	logger:info("Predictions written to file " .. predictedOutputsFileName)
end
