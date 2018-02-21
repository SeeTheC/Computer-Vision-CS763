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

--linear x y
--relu
--batchnorm
--convolution x y z a b c

strsplit=function (str)
   words = {}
   for word in str:gmatch("%w+") do table.insert(words, word) end
   return words;
end


print(opt["config"]);
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
print(weightPath);
print(biasPath);

