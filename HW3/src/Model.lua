require "Logger"
require "Linear"
require "ReLU"
require "Criterion"

local Linear = torch.class("Model")

function Model:__init()
	logger:debug("Initializing the Model")
	self.Layers = {}
	self.isTrain = false
end

function Model:addLayer(layer)
	table.insert(self.Layers, layer)
end

function Model:printModel()
	for i = 1, #self.Layers do
		print(torch.type(self.Layers[i]))
	end
end

function Model:forward(input)
	local nextInput = input
	for i = 1, #self.Layers do
		nextInput = self.Layers[i]:forward(nextInput)
	end
	return nextInput
end

function Model:backward(input, gradOutput, scale)
	scale = scale or 1
	local nextGradOutput = gradOutput
	for i = #self.Layers, 2, -1 do
		nextGradOutput = self.Layers[i]:backward(self.Layers[i - 1].output, nextGradOutput, scale)
	end
	nextGradOutput = self.Layers[1]:backward(input, nextGradOutput, scale)
	return nextGradOutput
end

function Model:dispGradParam()
	for i = #self.Layers, 1 do
		if torch.type(self.Layer[i]) == "Linear" then
			print(self.Layers[i].W)
			print(self.Layers[i].B)
		end
	end
end

function Model:clearGradParam()
	for i = 1, #self.Layers do
		self.Layers[i]:resetGrads()
	end
end

function Model:__tostring__()
	local string = "Model\n"
	string = string .. "[\n"
	for i = 1, #self.Layers do
		string = string .. "\t" .. string.format("(%d)\t-->\t%s", i, self.Layers[i]) .. "\n"
	end
	string = string .. "]"
	return string
end
