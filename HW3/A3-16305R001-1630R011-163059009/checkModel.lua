require "xlua"
require "Logger"

require "Linear"
require "ReLU"
require "BatchNormalization"
-- require "SpatialConvolution"
require "Model"

require "Criterion"
require "GradientDescent"


local cmd = torch.CmdLine();
if not opt then
	cmd:text()
	cmd:text('Options:')
	cmd:option("-config" ,"modelConfig_1.txt" ,"/path/to/modelConfig.txt")
	cmd:option("-i" ,"input.bin" ,"/path/to/input.bin")
	cmd:option("-ig" ,"gradInput.bin" ,"/path/to/gradInput.bin")
	cmd:option("-o" ,"output.bin" ,"/path/to/output.bin")
	cmd:option("-ow" ,"gradWeight.bin" ,"/path/to/gradWeight.bin")
	cmd:option("-ob" ,"gradB.bin" ,"/path/to/gradB.bin")
	cmd:option("-og" ,"gradOutput.bin" ,"/path/to/gradOutput.bin")

	cmd:text()
	opt = cmd:parse(arg or {})
end

strsplit=function (str)
	words = {}
	for word in str:gmatch("%w+") do table.insert(words, word) end
	return words;
end


-- Creating Model
local mlp = Model();
local weightPath="";
local biasPath="";
local lno=1;
for line in io.lines(opt["config"]) do
--	linear 192 10
--	relu
--	ba
--	mlp:addLayer(Liner(192, 10))
	if lno==1 then
		noOfLayers=tonumber(line);
	elseif lno <= 1 + noOfLayers then
		tbl=strsplit(line);
		if tbl[1] == 'linear' then
			mlp:addLayer(Linear(tonumber(tbl[2]),tonumber(tbl[3])));
		elseif tbl[1] == 'relu' then
			mlp:addLayer(ReLU());
		elseif tbl[1] == 'batchnorm' then
			mlp:addLayer(BatchNormalization())
		end
	elseif lno <= 1 + noOfLayers + 1 then
		weightPath=line;
	elseif lno <= 1 + noOfLayers + 2 then
		biasPath=line;
	end
	lno=lno+1;
end

local weights = torch.load(weightPath)
local biases = torch.load(biasPath)
local count = 1
for i = 1, #mlp.Layers do
	if torch.type(mlp.Layers[i]) == "Linear" then
		mlp.Layers[i].W = weights[count]
		mlp.Layers[i].B = biases[count]
		count = count + 1
	end
end
local input = torch.load(opt["i"])
local gradInput = torch.load(opt["ig"])

local output = mlp:forward(input:resize(5, 192))
local gradOutput = mlp:backward(input:resize(5, 192), gradInput)
local gradW = {}
local gradB = {}
for i = 1, #mlp.Layers do
	if torch.type(mlp.Layers[i]) == "Linear" then
		table.insert(gradW, mlp.Layers[i].gradW)
		table.insert(gradB, mlp.Layers[i].gradB)
	end
end
torch.save(opt["o"], output)
torch.save(opt["og"], gradOutput)
torch.save(opt["ow"], gradW)
torch.save(opt["ob"], gradB)

-- print(torch.sum(torch.load(opt["o"]) - output))
-- for i = 1, #gradW do
-- 	print(torch.sum(torch.load(opt["ow"])[i] - gradW[i]))
-- 	print(torch.sum(torch.load(opt["ob"])[i] - gradB[i]))
-- end
