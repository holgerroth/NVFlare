# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: grpcadapter.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x11grpcadapter.proto\x12\x0c\x66lower.proto\"\xbc\x01\n\x10MessageContainer\x12>\n\x08metadata\x18\x01 \x03(\x0b\x32,.flower.proto.MessageContainer.MetadataEntry\x12\x19\n\x11grpc_message_name\x18\x02 \x01(\t\x12\x1c\n\x14grpc_message_content\x18\x03 \x01(\x0c\x1a/\n\rMetadataEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\x32^\n\x0bGrpcAdapter\x12O\n\x0bSendReceive\x12\x1e.flower.proto.MessageContainer\x1a\x1e.flower.proto.MessageContainer\"\x00\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'grpcadapter_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _MESSAGECONTAINER_METADATAENTRY._options = None
  _MESSAGECONTAINER_METADATAENTRY._serialized_options = b'8\001'
  _MESSAGECONTAINER._serialized_start=36
  _MESSAGECONTAINER._serialized_end=224
  _MESSAGECONTAINER_METADATAENTRY._serialized_start=177
  _MESSAGECONTAINER_METADATAENTRY._serialized_end=224
  _GRPCADAPTER._serialized_start=226
  _GRPCADAPTER._serialized_end=320
# @@protoc_insertion_point(module_scope)
