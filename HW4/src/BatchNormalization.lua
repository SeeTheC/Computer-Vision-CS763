require "src.Logger"

local BatchNormalization = torch.class("BatchNormalization")

function BatchNormalization:__init()
	logger:debug("Initializing BatchNormalization Layer")
	self:resetGradsAndOutputs()
end

function BatchNormalization:resetGradsAndOutputs()
	self.output = torch.Tensor()
	self.gradInput = torch.Tensor()
end

function BatchNormalization:forward(input)
	local mean = torch.mean(input)
	local std = torch.std(input)
	self.output = (input - mean) / std
	return self.output
end

function BatchNormalization:backward(input, gradOutput)
	self.gradInput = gradOutput
	return self.gradInput
end

function BatchNormalization:updateParameters(learningRate)
end

function BatchNormalization:__tostring__()
	return torch.type(self)
end
