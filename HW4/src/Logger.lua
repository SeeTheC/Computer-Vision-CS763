local Logger = torch.class("Logger")

DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40
DISABLE = 100

function Logger:__init(logging_level, log_to_file)
	self.logging_level = logging_level or DISABLE

	self.log_to_file = (log_to_file == nil or log_to_file == true) and true or false
	if self.log_to_file then
		self.log_file_name = os.date("%Y-%m-%d-%X.log")
		print("Logging to file " .. self.log_file_name)
	end

	self:printLoggingLevel()
end

-- Changes the log level at run time
function Logger:setLevel(logging_level)
	self.logging_level = logging_level or DISABLE
	self:printLoggingLevel()
end

function Logger:printLoggingLevel()
	if self.logging_level == DISABLE then
		print("Logging is DISABLED")
	elseif self.logging_level == DEBUG then
		print("Logging level is set to DEBUG")
	elseif self.logging_level == INFO then
		print("Logging level is set to INFO")
	elseif self.logging_level == WARN then
		print("Logging level is set to WARN")
	elseif self.logging_level == ERROR then
		print("Logging level is set to ERROR")
	else
		print("Unexpected logging level")
		error()
	end
end

-- Debug log message
function Logger:debug(message)
	if self.logging_level <= DEBUG then
		print(os.date("%Y-%m-%d-%X") .. " [DEBUG] " .. tostring(message))
		if self.log_to_file then
			local log_file = io.open(self.log_file_name, "a")
			log_file:write(os.date("%Y-%m-%d-%X") .. " [DEBUG] " .. tostring(message) .. "\n")
			log_file:close()
		end
	end
end

-- Info log message
function Logger:info(message)
	if self.logging_level <= INFO then
		print(os.date("%Y-%m-%d-%X") .. " [INFO ] " .. tostring(message))
		if self.log_to_file then
			local log_file = io.open(self.log_file_name, "a")
			log_file:write(os.date("%Y-%m-%d-%X") .. " [INFO ] " .. tostring(message) .. "\n")
			log_file:close()
		end
	end
end

-- Warning log message
function Logger:warn(message)
	if self.logging_level <= WARN then
		print(os.date("%Y-%m-%d-%X") .. " [WARN] " .. tostring(message))
		if self.log_to_file then
			local log_file = io.open(self.log_file_name, "a")
			log_file:write(os.date("%Y-%m-%d-%X") .. " [WARN ] " .. tostring(message) .. "\n")
			log_file:close()
		end
	end
end

-- Error log message
function Logger:error(message)
	if self.logging_level <= ERROR then
		print(os.date("%Y-%m-%d-%X") .. " [ERROR] " .. tostring(message))
		if self.log_to_file then
			local log_file = io.open(self.log_file_name, "a")
			log_file:write(os.date("%Y-%m-%d-%X") .. " [ERROR] " .. tostring(message) .. "\n")
			log_file:close()
			error()
		end
	end
end

-- Generic log message
function Logger:log(message)
	print(os.date("%Y-%m-%d-%X") .. "[MSG  ]")
	print(tostring(message))
	if self.log_to_file then
		local log_file = io.open(self.log_file_name, "a")
		log_file:write(os.date("%Y-%m-%d-%X") .. " [MSG  ]\n")
		log_file:write(tostring(message) .. "\n")
		log_file:close()
	end
end

-- Initialize Logger object
logger = Logger.new(DEBUG, false)
