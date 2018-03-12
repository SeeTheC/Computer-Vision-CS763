require "src.Logger"

local BatchNormalization = torch.class("BatchNormalization")

function BatchNormalization:__init()
	logger:debug("Initializing BatchNormalization Layer")
	self.output = torch.Tensor()
	self.gradInput = torch.Tensor()
end

function BatchNormalization:forward(input)
	local mean = torch.mean(input)
	local std = torch.std(input)
	self.output = (input - mean) / std
end

function BatchNormalization:backward(input, gradOutput)
	self.gradInput = gradOutput
	return self.gradInput
end

function BatchNormalization:updateParameters(learningRate)
end
