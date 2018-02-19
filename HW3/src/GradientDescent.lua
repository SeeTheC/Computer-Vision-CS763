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
	if n_images % batch_size ~= 0 then
		logger:error("Input size " .. n_images .. " is not a multiple of batch size " .. batch_size)
		error()
	end
	local height = input:size(2)
	local width = input:size(3)

	while true do
		iteration = iteration + 1
		local criterion_output = 0
		local n_batches = n_images / batch_size
		local start_index = 1 + torch.Tensor(1):random(0, n_batches - 1)[1] * batch_size
		-- for batch = 1, n_batches do
			local batch_input = input:narrow(1, start_index, batch_size):resize(batch_size, height * width)
			local output = self.model:forward(batch_input)
			-- print(iteration, batch, torch.mean(torch.max(output, 2), 1)[1][1], torch.mean(torch.min(output, 2), 1)[1][1])
			criterion_output = criterion_output + self.criterion:forward(output, target)

			local criterion_gradInput = self.criterion:backward(output, target)
			self.model:backward(batch_input, criterion_gradInput, self.learningRate)
			start_index = start_index + batch_size
		-- end

		for i = 1, #self.model.Layers do
			self.model.Layers[i]:resetGrads()
		end

		criterion_output = criterion_output / n_batches
		if iteration then
			logger:info("Iteration: " .. iteration .. ", Loss: " .. criterion_output)
		end

		if self.maxIterations > -1 and self.maxIterations < iteration then
			logger:info("Maximum iterations reached in Gradient Descent. Loss = " .. criterion_output)
			return criterion_output
		end

		if math.abs(previousLoss - criterion_output) < self.epsilon then
			logger:info("Gradient Descent converged with loss = " .. criterion_output)
			return criterion_output
		end
		previousLoss = criterion_output
	end
end
