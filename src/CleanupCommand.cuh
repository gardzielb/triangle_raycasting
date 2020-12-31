//
// Created by bartosz on 22.11.2020.
//

#ifndef FISH_CLEANUPCOMMAND_CUH
#define FISH_CLEANUPCOMMAND_CUH

#include <vector>
#include <algorithm>

// a class that encapsulates memory free requests and allows client to execute them when necessary
class CleanupCommand
{
public:
	virtual void execute() = 0;

	virtual ~CleanupCommand() = default;
};


template<typename T>
class HostCleanupCommand : public CleanupCommand
{
private:
	T * array = nullptr;

public:
	explicit HostCleanupCommand( T * array )
	{
		this->array = array;
	}

	void execute() override
	{
		delete array;
	}
};


template<typename T>
class GpuCleanupCommand : public CleanupCommand
{
private:
	T * array = nullptr;

public:
	explicit GpuCleanupCommand( T * array )
	{
		this->array = array;
	}

	void execute() override
	{
		cudaFree( array );
	}
};


class CompositeCleanupCommand : public CleanupCommand
{
private:
	std::vector<CleanupCommand *> commands;

public:
	explicit CompositeCleanupCommand( std::vector<CleanupCommand *> commands )
			: commands( std::move( commands ) )
	{}

	void execute() override
	{
		std::for_each( commands.begin(), commands.end(), []( CleanupCommand * cmd ) { cmd->execute(); } );
	}
};


#endif //FISH_CLEANUPCOMMAND_CUH
