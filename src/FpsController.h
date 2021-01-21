//
// Created by bartosz on 30.11.2020.
//

#ifndef FISH_SHOAL_ANIMATION_FPSCONTROLLER_H
#define FISH_SHOAL_ANIMATION_FPSCONTROLLER_H

#include "dependencies/helper_timer.h"
#include <cstdio>
#include <algorithm>

class FpsController
{
private:
	int fpsCount = 0;        // FPS count for averaging
	int fpsLimit = 1;        // FPS limit for sampling
	float avgFPS = 0.0f;
	unsigned int frameCount = 0;
	StopWatchInterface * timer;

public:
	FpsController()
	{
		sdkCreateTimer( &timer );
	}

	void startTimer()
	{
		sdkStartTimer( &timer );
	}

	std::string computeFPS()
	{
		sdkStopTimer( &timer );
		frameCount++;
		fpsCount++;

		if ( fpsCount == fpsLimit )
		{
			avgFPS = 1.f / (sdkGetAverageTimerValue( &timer ) / 1000.f);
			fpsCount = 0;
			fpsLimit = (int) std::max( avgFPS, 1.f );

			sdkResetTimer( &timer );
		}

		char fps[256];
		sprintf( fps, "%3.1f FPS", avgFPS );
		return std::string( fps );
	}
};


#endif //FISH_SHOAL_ANIMATION_FPSCONTROLLER_H
