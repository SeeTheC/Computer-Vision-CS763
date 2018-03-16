require "src.Logger"
require "src.Helper"
require "src.Model"
require "src.RNN"
require "src.BatchNormalization"
require "src.Linear"

local cmd = torch.CmdLine()
cmd:text()
cmd:text("Options:")
cmd:option("-modelName", "", "/path/to/model")
cmd:option("-data", "", "/path/to/input")
cmd:text()
opt = cmd:parse(arg or {})

local modelName = opt["modelName"]
local inputFileName = opt["data"]

if modelName == "" or inputFileName == "" then
	cmd:help()
	os.exit()
end

-- Load and processing data
local inputMappings, countInput = loadMappings("src/input_mappings")
local outputMappings, countOutput = loadMappings("src/output_mappings")
local inputs = oneHotEncode(inputFileName, inputMappings, countInput)

local modelLoadPath = modelName .. "/model.bin"

if modelLoadPath then
	logger:info("Load Model Path: " .. modelLoadPath)
	model = torch.load(modelLoadPath)
	local predictedOutputsFileName = "testPredictions.csv"
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
