#include "caffe/vdnn.hpp"

DeviceMemoryManager dmm;

DeviceMemoryManager::DeviceMemoryManager() {
	vmt.UsedSize = 0;
	vmt.DeviceSize = 1000000000;
}

DeviceMemoryManager::~DeviceMemoryManager() {

}

void DeviceMemoryManager::registerMemoryEntry(SyncedMemory *data) {
	VirtualMemoryEntry *entry = new VirtualMemoryEntry;
	entry->data = data;
	entry->gpu_ptr = NULL;
	entry->size = data->size(); // always 0
	vmt.MemoryTable.push_back(*entry);

	//printf("\n\nRegister memory!\n\n");
}

void DeviceMemoryManager::unregisterMemoryEntry(SyncedMemory *data) {
	vmtIterator it;
	for (it=vmt.MemoryTable.begin(); it<vmt.MemoryTable.end(); it++) {
		if ( (*it).data == data ) {
			vmt.MemoryTable.erase(it);
			//printf("\n\nfound & unregister memory!\n\n");
			return;
		}
	}
}

bool DeviceMemoryManager::deallocate(size_t size) {
	vmtIterator it = vmt.MemoryTable.begin();
	//vmt.UsedSize += size;
	while (vmt.UsedSize+size >= vmt.DeviceSize
		&& it<vmt.MemoryTable.end()) {
		if ( (*it).gpu_ptr!=NULL && (*it).size >= THRESH && hit.end()==hit.find((*it).gpu_ptr) ) {
			if ( !(*it).data->own_gpu_data() ) {
				printf("Skip GPU memory owned by others\n");
				//getchar();
				continue;
			}
			//printf("cudaFree size %d\n", (*it).data->size());
			(*it).data->reset_gpu_data();
			//printf("In Table, cudaFree(%p) with size %d\n", (*it).gpu_ptr, (*it).size);
			//cudaDeviceSynchronize();
			CUDA_CHECK(cudaFree((*it).gpu_ptr));
			//printf("cudaFree success!\n");
			//printf("Before free UsedSize : %d\n", vmt.UsedSize);
			(*it).gpu_ptr = NULL;
			vmt.UsedSize -= (*it).size;
			if ( vmt.UsedSize+size < vmt.DeviceSize )	
				return true;
			//printf("UsedSize : %d\n", vmt.UsedSize);
		}
		++it;
		//printf("move to next entry\n");
	}
	dumpMemoryTable();
	printf("!!! Can't find any chunk for deallocation!!!\n");
	getchar();
	return false;
}

void DeviceMemoryManager::checkAvailMem(size_t size) {
	//printf("Need memory chunk with size %d!\n", size);
	if (vmt.UsedSize + size >= vmt.DeviceSize) {
		//printf("used: %d need deallocation\n", vmt.UsedSize);
		bool flag = deallocate(size);
		//printf("!!!!!%d!!!!\n", flag);
		//printf("used: %d after deallocation\n", vmt.UsedSize);
	} else {
		//printf("Safe!\n");
	}
	//printf("%p used: %d available: %d\n", this, vmt.UsedSize, vmt.DeviceSize-vmt.UsedSize);
	//dumpMemoryTable();
}

VirtualMemoryEntry *DeviceMemoryManager::lookupMemoryEntry(SyncedMemory *data) {
	vmtIterator it;
	for (it=vmt.MemoryTable.begin(); it<vmt.MemoryTable.end(); it++) {
		if ( (*it).data == data ) {
			//printf("In the table: gpu_ptr: %p size: %d\n", (*it).gpu_ptr, (*it).size);
			//printf("Actual data: size: %d\n", data->size());
			return &(*it);
		}
	}
	return NULL;
}

void DeviceMemoryManager::updateMemoryEntry(VirtualMemoryEntry * me) {
	


}

void DeviceMemoryManager::updateMallocMem(SyncedMemory *data, void *gpu_ptr, size_t size) {
	VirtualMemoryEntry *entry = lookupMemoryEntry(data);
	char c;
	if (entry==NULL)	std::cin >> c;
	entry->gpu_ptr = gpu_ptr;
	entry->size = size;
	vmt.UsedSize += size;
	//printf("Add %d of GPU memory %p to MemoryTable\n", size, gpu_ptr);
	//dumpMemoryTable();
	//getchar();
}

void DeviceMemoryManager::dumpMemoryTable() {
	vmtIterator it;
	int i = 0;
	for (it=vmt.MemoryTable.begin(); it<vmt.MemoryTable.end(); it++, i++) {
		if ( (*it).gpu_ptr!=NULL ) 
			printf("Entry: %d: SyncM %p - gpu_ptr (%d) %p Size %lu\n", i, (*it).data, (*it).data->own_gpu_data(),(*it).gpu_ptr, (*it).size);
	}
}

bool DeviceMemoryManager::isInUse(void *gpu_ptr) {
	//queue<void *>::iterator it = find(hit.begin(), hit.end(), gpu_ptr);
	//return it!=hit.end();
	return false;
}

void DeviceMemoryManager::resetHitTable() {
	//printf("Before clear: size = %d\n", hit.size());
	hit.clear();
}

void DeviceMemoryManager::insertHitTable(void *gpu_ptr) {
	hit.insert(gpu_ptr);
}
/*
void DeviceMemoryManager::updateHitTable(void *gpu_ptr) {
	if (!isInUse(gpu_ptr) && )

}*/
