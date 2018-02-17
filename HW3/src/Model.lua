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

function Model:backward(input, gradOutput)
	local nextGradOutput = gradOutput
	for i = #self.Layers, 2, -1 do
		nextGradOutput = self.Layers[i]:backward(self.Layers[i - 1].output, nextGradOutput)
		if torch.type(self.Layers[i]) ~= "ReLU" then
			self.Layers[i]:updateParams(0.001)
		end
	end
	nextGradOutput = self.Layers[1]:backward(input, nextGradOutput)
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
