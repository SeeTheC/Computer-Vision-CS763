function loadMappings(mappingFile)
	local mappings = {}
	local count = 0
	local file = io.open(mappingFile)
	if file then
		for line in file:lines() do
			local value, key = unpack(line:split("\t"))
			mappings[tonumber(key)] = tonumber(value)
			count = count + 1
		end
	end
	return mappings, count
end

function oneHotEncodeAux(vars, mappings, count)
	local oneHots = torch.zeros(count, #vars)
	for i = 1, #vars do
		oneHots[mappings[tonumber(vars[i])]][i] = 1
	end
	return oneHots
end

function oneHotEncode(filename, mappings, count)
	local input = {}
	local file = io.open(filename)
	if file then
		for line in file:lines() do
			table.insert(input, oneHotEncodeAux(line:split(" "), mappings, count))
		end
	end
	return input
end
