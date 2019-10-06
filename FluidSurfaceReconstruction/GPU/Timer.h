#pragma once

#include <chrono>
#include <memory>

//! Simple timer class.
class Timer
{
public:
	typedef std::shared_ptr<Timer> ptr;

	//! Constructs the timer and start ticking.
	Timer();

	//! Returns the time duration since the creation or reset in seconds.
	double durationInSeconds() const;
	
	//! Returns the time duration since the creation or reset in milliseconds.
	double durationInMilliseconds() const;

	//! Resets the timer.
	void reset();

private:
	std::chrono::steady_clock _clock;
	std::chrono::steady_clock::time_point _startingPoint;
};
