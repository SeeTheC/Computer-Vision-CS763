require "xlua"
cmd = torch.CmdLine()

local cmd = torch.CmdLine()
if not opt then
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Options:')
   cmd:option("-modelName" ,"" ,"/path/to/<model>")
   cmd:option("-data" ,"test.bin" ,"/path/to/test.bin (test.bin is a 4D tensor)")
   cmd:option("-target" ,"testPrediction.bin" ,"/path/to/target/testPrediction.bin")
   cmd:text()
   opt = cmd:parse(arg or {})
end


function predict()
	local arch = torch.load(opt["modelName"].."/model.bin");
	local model= arch[1];
	local testData = torch.load(opt["data"]):double()
	local output = model:forward(testData:resize(testData:size(1), testData:size(2) * testData:size(3)))
	local value, index = torch.max(output, 2)
	local indexfile = io.open(opt["target"], "w")
	indexfile:write("id,label\n")
	for i = 1, value:size(1) do
		indexfile:write(i - 1 .. "," .. index[i][1] - 1 .. "\n")
	end
	io.close(indexfile)
end
predict();
