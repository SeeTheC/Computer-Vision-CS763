require "Logger"

local GradientDescent = torch.class("GradientDescent")

function GradientDescent:__init(model, criterion, learningRate, maxIterations)
	self.model = model
	self.criterion = criterion
	self.learningRate = learningRate or 0
	self.maxIterations = maxIterations or -1
	self.epsilon = 1e-6
end

function GradientDescent:train(input, target, batch_size)
	logger:debug("Beginning Gradient Descent")
	local iteration = 0
	local previousLoss = 1e5

	local n_images = input:size(1)
	batch_size = batch_size or n_images
	local height = input:size(2)
	local width = input:size(3)

	local trainRatio = 0.8
	local validationRatio = 1 - trainRatio
	local trainInput = input:narrow(1, 1, n_images * trainRatio)
	local trainOutput = target:narrow(1, 1, n_images * trainRatio)
	local validationInput = input:narrow(1, n_images * trainRatio + 1, n_images * validationRatio)
	local validationOutput = target:narrow(1, n_images * trainRatio + 1, n_images * validationRatio)

	while true do
		iteration = iteration + 1
		local permutations = torch.randperm(trainInput:size(1)):long():narrow(1, 1, batch_size)

		local batch_input = trainInput:index(1, permutations)
		local batch_output = trainOutput:index(1, permutations)

		local output = self.model:forward(batch_input)

		local criterion_output = self.criterion:forward(output, batch_output)
		local criterion_gradInput = self.criterion:backward(output, batch_output)

		self.model:backward(batch_input, criterion_gradInput, self.learningRate)

		-- for i = 1, #self.model.Layers do
		-- 	self.model.Layers[i]:resetGrads()
		-- end
		local x = self.model:forward(validationInput)
		print(x)
		local _, predictionIndex = torch.max(x, 2)
		local accuracy = torch.sum((predictionIndex:double() - validationOutput):apply(function(p) return (p == 0 and 1 or 0) end))
		if iteration then
			logger:info("Iteration: " .. iteration .. " Loss: " .. string.format("%0.5f", criterion_output) .. " Accuracy: " .. string.format("%0.3f", accuracy / validationOutput:size(1) * 100) .. "%")
		end

		if self.maxIterations > -1 and self.maxIterations <= iteration then
			logger:info("Maximum iterations reached in Gradient Descent. Loss = " .. criterion_output)
			return criterion_output
		end

		-- if math.abs(previousLoss - criterion_output) < self.epsilon then
		-- 	logger:info("Gradient Descent converged with loss = " .. criterion_output)
		-- 	return criterion_output
		-- end
		-- previousLoss = criterion_output
	end
end
