require "Logger"

local Criterion = torch.class('Criterion')

function Criterion:__init ()
	logger:debug("Initializing Criterion Layer")
end

function Criterion:softmax(X)
	exps = torch.exp(X)
	return exps / torch.sum(exps)
end

function Criterion:forward(input, target)
	m = target:size(1)
	inputSize = input:size(1)
	local loss = 0
	for i = 1, inputSize do
		p = self:softmax(input[i])
		--here torch.max just use to extract value from the tensor
		loss = loss - torch.log(p[torch.max(target[i])])
	end
	return loss / m
end

function Criterion:backward(input, target)
	inputSize = input:size(1)
	local gradInput = torch.zeros(input:size(1), input:size(2))
	for i = 1, inputSize do
		p = self:softmax(input[i])
		local max = torch.max(target[i])
		p[max] = p[max] - 1
		gradInput[i] = p
	end
	return gradInput
end
