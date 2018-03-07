require "Logger"

local Criterion = torch.class("Criterion")

function Criterion:__init ()
	logger:debug("Initializing Criterion Layer")
	self.output = 0
end

function Criterion:softmax(X)
	-- Subtracting the max to avoid overflows
	exps = torch.exp(X - torch.max(X))
	if X:nDimension() == 1 then
		return exps / torch.sum(exps)
	end

	for i = 1, exps:size(1) do
		exps[i] = exps[i] / torch.sum(exps[i])
	end
	return exps
end

function Criterion:forward(input, target)
	local output = torch.log(self:softmax(input))
	if input:nDimension() == 1 then
		self.output = -output[target[1]]
		return self.output
	end

	for i = 1, input:size(1) do
		self.output = self.output - output[i][target[i]]
	end
	self.output = self.output / input:size(1)
	return self.output
end

function Criterion:backward(input, target)
	local gradInput = self:softmax(input)
	if input:nDimension() == 1 then
		gradInput[target[1]] = gradInput[target[1]] - 1
		return gradInput
	end

	for i = 1, input:size(1) do
		gradInput[i][target[i]] = gradInput[i][target[i]] - 1
	end
	return gradInput / input:size(1)
end
