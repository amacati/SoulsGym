/* Code adapted from https://github.com/IamSanjid/ce_speed_hack */
#include <Windows.h>
#include "detours.h"
// Detours: https://www.microsoft.com/en-us/research/project/detours/
#include <chrono>
#include <thread>
#include <mutex>

#pragma comment(lib,"detours.lib")	   // Need to include this so we can use Detours
#pragma comment(lib,"Kernel32.lib")	   // Need to include this since we're hooking QueryPerformanceCounter and GetTickCount which reside inside the Kernel32 library
#pragma comment(lib,"Winmm.lib")	   // Neet to include this since we're hooking timeGetTime which resides inside the Winmm library

#define BUFSIZE 512

extern"C" {
	static BOOL(WINAPI* originalQueryPerformanceCounter)(LARGE_INTEGER* performanceCounter) = QueryPerformanceCounter;
	static DWORD(WINAPI* originalGetTickCount)() = GetTickCount;
	static ULONGLONG(WINAPI* originalGetTickCount64)() = GetTickCount64;
	static DWORD(WINAPI* originalTimeGetTime)() = timeGetTime;
}

std::recursive_mutex GTCMutex;
std::recursive_mutex QPCMutex;

template<class T>
class SpeedHackClass
{
private:
	double speed = 0;
	T initialoffset;
	T initialtime;
public:
	SpeedHackClass()
	{
		speed = 1.0;
	}
	SpeedHackClass(T _initialtime, T _initialoffset, double _speed = 1.0)
	{
		speed = _speed;
		initialoffset = _initialoffset;
		initialtime = _initialtime;
	}

	double get_speed() const { return speed; }

	T get(T currentTime)
	{
		T false_val = (T)((currentTime - initialtime) * speed) + initialoffset;
		return (T)false_val;
	}

	void set_speed(double _speed)
	{
		speed = _speed;
	}

	void set_offsets(T _initialtime, T _initialoffset)
	{
		initialtime = _initialtime;
		initialoffset = _initialoffset;
	}
};


SpeedHackClass<LONGLONG> h_QueryPerformanceCounter;
SpeedHackClass<DWORD> h_GetTickCount;
SpeedHackClass<ULONGLONG> h_GetTickCount64;
SpeedHackClass<DWORD> h_GetTime;

// QueryPerformanceCounter is generally what is used to calculate how much time has passed between frames. It will set the performanceCounter to the amount of micro seconds the machine has been running
// https://msdn.microsoft.com/en-us/library/windows/desktop/ms644904(v=vs.85).aspx

BOOL WINAPI newQueryPerformanceCounter(LARGE_INTEGER* counter) {
	std::lock_guard<std::recursive_mutex> qpc_lock(QPCMutex);
	LARGE_INTEGER currentLi;
	LARGE_INTEGER falseLi;
	originalQueryPerformanceCounter(&currentLi);
	falseLi.QuadPart = h_QueryPerformanceCounter.get(currentLi.QuadPart);
	*counter = falseLi;
	return true;
}

// GetTickCount can also be used to calculate time between frames, but is used less since it's less accurate than QueryPerformanceCounter
// https://msdn.microsoft.com/en-us/library/windows/desktop/ms724408%28v=vs.85%29.aspx

DWORD WINAPI newGetTickCount() {
	std::lock_guard<std::recursive_mutex> gtc_lock(GTCMutex);
	return h_GetTickCount.get(originalGetTickCount());;																					// Return false tick count
}

// GetTickCount64 can also be used to calculate time between frames, but is used less since it's less accurate than QueryPerformanceCounter
//https://docs.microsoft.com/en-us/windows/desktop/api/sysinfoapi/nf-sysinfoapi-gettickcount64

ULONGLONG WINAPI newGetTickCount64() {
	std::lock_guard<std::recursive_mutex> gtc_lock(GTCMutex);
	return h_GetTickCount64.get(originalGetTickCount64());
}

// timeGetTime can also be used to caluclate time between frames, as with GetTickCount it isn't as accurate as QueryPerformanceCounter
// https://msdn.microsoft.com/en-us/library/windows/desktop/dd757629(v=vs.85).aspx

DWORD WINAPI newTimeGetTime() {
	return h_GetTime.get(originalTimeGetTime());
}

LARGE_INTEGER initialtime64;
LARGE_INTEGER initialoffset64;

//Called by createremotethread
void InitializeSpeedHackConnection(LPVOID hModule) {
	float speed = 1.0;
	{
		std::unique_lock<std::recursive_mutex> qpc_lock(QPCMutex);
		std::unique_lock<std::recursive_mutex> gtc_lock(GTCMutex);
		originalQueryPerformanceCounter(&initialtime64);
		newQueryPerformanceCounter(&initialoffset64);
		h_QueryPerformanceCounter = SpeedHackClass<LONGLONG>(initialtime64.QuadPart, initialoffset64.QuadPart, speed);
		h_GetTickCount = SpeedHackClass<DWORD>(originalGetTickCount(), newGetTickCount(), speed);
		h_GetTickCount64 = SpeedHackClass<ULONGLONG>(originalGetTickCount64(), newGetTickCount64(), speed);
		h_GetTime = SpeedHackClass<DWORD>(originalTimeGetTime(), newTimeGetTime(), speed);
	}

	HANDLE hPipe;
	DWORD dwRead;
	union {
		float float_buffer[BUFSIZE];
		char byte_buffer[BUFSIZE * 4];
	} u;
	int float_idx = 0;
	float speed_cmd = 1.0;

	hPipe = CreateNamedPipe(TEXT("\\\\.\\pipe\\DS3SpeedHackPipe"),
		PIPE_ACCESS_DUPLEX,
		PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,   // FILE_FLAG_FIRST_PIPE_INSTANCE is not needed but forces CreateNamedPipe(..) to fail if the pipe already exists...
		1,
		1024 * 16,
		1024 * 16,
		NMPWAIT_USE_DEFAULT_WAIT,
		NULL);
	while (hPipe != INVALID_HANDLE_VALUE)
	{
		if (ConnectNamedPipe(hPipe, NULL) != FALSE)   // wait for someone to connect to the pipe
		{
			while (ReadFile(hPipe, u.byte_buffer, sizeof(u.byte_buffer) - 1, &dwRead, NULL) != FALSE)
			{
				float_idx = dwRead / 4 - 1;
				speed_cmd =  u.float_buffer[float_idx];
				if (speed_cmd >= 0.)
				{
					std::unique_lock<std::recursive_mutex> qpc_lock(QPCMutex);
					std::unique_lock<std::recursive_mutex> gtc_lock(GTCMutex);
					// Update Query Counter
					originalQueryPerformanceCounter(&initialtime64);
					newQueryPerformanceCounter(&initialoffset64);
					h_QueryPerformanceCounter.set_offsets(initialtime64.QuadPart, initialoffset64.QuadPart);
					h_QueryPerformanceCounter.set_speed(speed_cmd);
					// Update Tick Counter
					h_GetTickCount.set_offsets(originalGetTickCount(), newGetTickCount());
					h_GetTickCount.set_speed(speed_cmd);
					// Update Tick Count 64
					h_GetTickCount64.set_offsets(originalGetTickCount64(), newGetTickCount64());
					h_GetTickCount64.set_speed(speed_cmd);
					// Update Get Time
					h_GetTime.set_offsets(originalTimeGetTime(), newTimeGetTime());
					h_GetTime.set_speed(speed_cmd);
				}
			}
		}
		DisconnectNamedPipe(hPipe);
	}
}

// This should be called when the DLL is Injected. You should call this in a new Thread.
void InitDLL(LPVOID hModule)
{
	{
		std::unique_lock<std::recursive_mutex> qpc_lock(QPCMutex);
		std::unique_lock<std::recursive_mutex> gtc_lock(GTCMutex);

		// Set initial values for hooked calculations
		originalQueryPerformanceCounter(&initialtime64);
		initialoffset64 = initialtime64;

		h_QueryPerformanceCounter = SpeedHackClass<LONGLONG>(initialtime64.QuadPart, initialoffset64.QuadPart);
		h_GetTickCount = SpeedHackClass<DWORD>(originalGetTickCount(), originalGetTickCount());
		h_GetTickCount64 = SpeedHackClass<ULONGLONG>(originalGetTickCount64(), originalGetTickCount64());
		h_GetTime = SpeedHackClass<DWORD>(originalTimeGetTime(), originalTimeGetTime());
	}
	// ah detours; they are awesome!!
	DisableThreadLibraryCalls((HMODULE)hModule);
	DetourTransactionBegin();
	DetourUpdateThread(GetCurrentThread());
	DetourAttach(&(PVOID&)originalQueryPerformanceCounter, newQueryPerformanceCounter);
	DetourAttach(&(PVOID&)originalGetTickCount, newGetTickCount);
	DetourAttach(&(PVOID&)originalGetTickCount64, newGetTickCount64);
	DetourAttach(&(PVOID&)originalTimeGetTime, newTimeGetTime);
	DetourTransactionCommit();
}

extern "C" __declspec(dllexport)BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID lpReserved)
{
	switch (ul_reason_for_call)
	{
	case DLL_PROCESS_ATTACH:
		CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)InitDLL, NULL, 0, NULL);		// Detours the 3 functions, enabling the speed hack
		CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)InitializeSpeedHackConnection, (LPVOID)hModule, 0, NULL);
		break;
	case DLL_PROCESS_DETACH:
		break;
	case DLL_THREAD_ATTACH:
		break;
	case DLL_THREAD_DETACH:
		break;
	}
	return TRUE;
}
