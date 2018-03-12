local Module = torch.class("Module")

function Module:__init()
	self.gradInput = torch.Tensor()
	self.output = torch.Tensor()
end

function Module:parameters()
	if self.W and self.B then
		return {self.W, self.B}, {self.gradW, self.gradB}
	elseif self.W then
		return {self.W}, {self.gradW}
	elseif self.bias then
		return {self.B}, {self.gradB}
	else
		return
	end
end

function Module:updateOutput(input)
	return self.output
end

function Module:forward(input)
	return self:updateOutput(input)
end

function Module:backward(input, gradOutput, scale)
	scale = scale or 1
	self:updateGrads(input, gradOutput, scale)
	return self.gradInput
end

function Module:updateGrads(input, gradOutput, scale)
	return self.gradInput
end

function Module:zeroGradParameters()
	local _, gradParams = self:parameters()
	if gradParams then
		for i = 1, #gradParams do
			gradParams[i]:zero()
		end
	end
end

function Module:updateParameters(learningRate)
	local params, gradParams = self:parameters()
	if params then
		for i = 1, #params do
			params[i]:add(-learningRate, gradParams[i])
		end
	end
end

function Module:training()
	self.train = true
end

function Module:evaluate()
	self.train = false
end
