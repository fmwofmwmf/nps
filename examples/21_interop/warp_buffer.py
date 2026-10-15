import uipc
import uipc.adapter.warp
import warp as wp

wp.init()

# create uipc buffer (managed by warp)
wb = uipc.adapter.warp.buffer(size=100, dtype=wp.float32, device="cuda")
print(wb.buffer_view())
wb.resize(10)
print(wb.buffer_view())

# get warp array from uipc buffer
print(wb.warp())
