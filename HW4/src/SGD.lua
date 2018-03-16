require "src.Logger"

local SGD = torch.class("SGD")

function SGD:__init(model, criterion, epochs, learningRate, trainInputs, trainTargets, validationInputs, validationTargets, accuracyAfterEpochs, saveModelAfterEpochs, saveModelPathPrefix)
	self.model = model
	self.criterion = criterion
	self.epochs = epochs
	self.learningRate = learningRate
	self.trainInputs = trainInputs
	self.trainTargets = trainTargets
	self.validationInputs = validationInputs
	self.validationTargets = validationTargets
	self.accuracyAfterEpochs = accuracyAfterEpochs or 100
	self.saveModelAfterEpochs = (saveModelAfterEpochs ~= nil or saveModelAfterEpochs <= self.epochs) and saveModelAfterEpochs or self.epochs
	self.saveModelPathPrefix = saveModelPathPrefix or os.date("rnn_%Y-%m-%d-%X")
end

function SGD:train()
	torch.manualSeed(os.time())
	for epoch = 1, self.epochs do
		local i = torch.random(1, #self.trainInputs)
		self.model:resetGradsAndOutputs()
		self.model:forward(self.trainInputs[i])
		self.criterion:forward(self.model.output, self.trainTargets[i])
		self.criterion:backward(self.model.output, self.trainTargets[i])
		self.model:backward(self.trainInputs[i], self.criterion.gradInput)
		self.model:updateParameters(self.learningRate)

		if epoch % self.accuracyAfterEpochs == 0 then
			local accuracy = self:accuracy()
			logger:info(string.format("Epochs: %4d, Accuracy: %0.5f", epoch, accuracy))
		end

		if epoch % self.saveModelAfterEpochs == 0 then
			local modelSavePath = self.saveModelPathPrefix .. os.date("model_%Y-%m-%d-%X_") .. epoch .. ".bin"
			logger:info("Model Save Path: " .. modelSavePath)
			torch.save(modelSavePath, self.model)
			-- Create a copy of the model so that it can be used at test time with a uniform name
			torch.save(self.saveModelPathPrefix .. "model.bin", self.model)
		end
	end
	logger:debug("SGD completed " .. self.epochs .. " epochs")
end

function SGD:accuracy()
	local correctPredictions = 0
	for k = 1, #self.validationInputs do
		self.model:resetGradsAndOutputs()
		self.model:predict(self.validationInputs[k])
		-- print(self.model.output[1][1], self.model.output[2][1])
		local _, predicted = torch.max(self.model.output, 1)
		local _, actual = torch.max(self.validationTargets[k], 1)

		if (predicted[1][1] == actual[1][1]) then
			correctPredictions = correctPredictions + 1
		end
	end
	local accuracy = correctPredictions / #self.validationInputs
	return accuracy
end
