require "src.Logger"

local Criterion = torch.class("Criterion")

function Criterion:__init ()
	logger:debug("Initializing Criterion Layer")
	self.output = 0
	self.gradInput = torch.Tensor()
end

function Criterion:softmax(X)
	-- Subtracting the max to avoid overflows
	exps = torch.exp(X - torch.max(X))
	return exps / torch.sum(exps)
end

function Criterion:forward(input, target)
	self.output = torch.max(-target:t() * torch.log(self:softmax(input)))
	return self.output
end

function Criterion:backward(input, target)
	self.gradInput = torch.cmul(self:softmax(input) - 1, target)
	return self.gradInput
end

function Criterion:updateParameters(learningRate)
end
