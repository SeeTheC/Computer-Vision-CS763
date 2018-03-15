require "src.Logger"

local Model = torch.class("Model")

function Model:__init()
	self.layers = {}
	self.train = false
end

function Model:resetGradsAndOutputs()
	for i = 1, #self.layers do
		self.layers[i]:resetGradsAndOutputs()
	end
end

function Model:add(layer)
	table.insert(self.layers, layer)
end

function Model:forward(input)
	local nextInput = input
	for i = 1, #self.layers do
		nextInput = self.layers[i]:forward(nextInput)
	end
	self.output = nextInput
	return self.output
end

function Model:predict(input)
	return self:forward(input)
end

function Model:backward(input, gradOutput)
	local nextGradOutput = gradOutput
	for i = #self.layers, 2, -1 do
		nextGradOutput = self.layers[i]:backward(self.layers[i - 1].output, nextGradOutput)
	end
	nextGradOutput = self.layers[1]:backward(input, nextGradOutput)
	self.gradInput = nextGradOutput
	return self.gradInput
end

function Model:updateParameters(learningRate)
	for i = 1, #self.layers do
		self.layers[i]:updateParameters(learningRate)
	end
end

function Model:__tostring__()
	local string = "Model\n"
	string = string .. "[\n"
	for i = 1, #self.Layers do
		string = string .. "\t" .. string.format("(%d) -->\t%s", i, self.Layers[i]) .. "\n"
	end
	string = string .. "]"
	return string
end
