from driftguard.federate.server.fed_server import FedServer, FedServerArgs, serve_fed_server
from driftguard.rpc.proxy import DataServiceProxy, ServerProxy
from driftguard.rpc.rpc import Node

data_node = Node("http://127.0.0.1", 12099)
fedserver_node = Node("0.0.0.0", 12000)
num_clients = 30

fedserver_args = FedServerArgs(
    data_service_proxy = DataServiceProxy(data_node),
    num_clients=num_clients,
)

# serve_fed_server(fedserver_node, fedserver_args)

server_proxy = ServerProxy(fedserver_node)
server_proxy.req_upload_obs((0, b"observation_data"))

# server_proxy.req_adv_step((0,))