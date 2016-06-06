#ifndef VDNN_HPP
#define VDNN_HPP

#include <cstdio>
#include <cstdlib>
#include <map>
#include <vector>
#include <queue>

#include <cuda.h>
#include <cuda_runtime.h>

#include "caffe/syncedmem.hpp"

/*
 * 0 - IsAllocated (Host)
 * 1 - IsAllocated (Device)
 * 2 - Dirty bit   (Host)
 * 3 - Dirty bit   (Device)
 */

#define VMM_NULL                0x00000000
#define VMM_HOST_ALLOCATED      0x00000001
#define VMM_DEVICE_ALLOCATED    0x00000002
#define VMM_HOST_DIRTY          0x00000004
#define VMM_DEVICE_DIRTY        0x00000008

#define THRESH			500000

using namespace std;
using namespace caffe;

typedef struct {
	SyncedMemory *data;
	void *gpu_ptr;
	size_t size;
} VirtualMemoryEntry; /* Virtual Memory Entry */

typedef struct {
	vector<VirtualMemoryEntry> MemoryTable;
	size_t DeviceSize;
	size_t UsedSize;
} VirtualMemoryTable;

typedef vector<VirtualMemoryEntry>::iterator vmtIterator;

class DeviceMemoryManager {
public:
	/*
	 * These are called by main module for instantiation of this class.
	 * See 'management.cpp' for more information.
	 */

	DeviceMemoryManager();
	~DeviceMemoryManager();

	void registerMemoryEntry(SyncedMemory *data);
	void unregisterMemoryEntry(SyncedMemory *data);

	/***************************
	 * Virtual Memory Allocation
	 ***************************/ 
	bool deallocate(size_t size);

	/***************************
	 * Virtual Memory Manipulation
	 **************************/
	void checkAvailMem(size_t size);

	VirtualMemoryEntry *lookupMemoryEntry(SyncedMemory *data);

	void updateMemoryEntry(VirtualMemoryEntry *me);
	void updateMallocMem(SyncedMemory *data, void *gpu_ptr, size_t size);

	void dumpMemoryTable();
	bool isInUse(void *gpu_ptr);

	void resetHitTable();
	void insertHitTable(void *gpu_ptr);

private:
	VirtualMemoryTable vmt;
	set<void *> hit;
};

extern DeviceMemoryManager dmm;

#endif
