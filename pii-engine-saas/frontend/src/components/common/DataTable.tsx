import React from "react";
import { Table, Empty, TableProps } from "antd";
import type { ColumnsType, TablePaginationConfig } from "antd/es/table";
import type { FilterValue, SorterResult } from "antd/es/table/interface";

export interface DataTableProps<T extends object> {
  columns: ColumnsType<T>;
  dataSource: T[];
  loading?: boolean;
  pagination?: TablePaginationConfig | false;
  rowKey?: string | ((record: T) => string);
  rowSelection?: TableProps<T>["rowSelection"];
  onChange?: (
    pagination: TablePaginationConfig,
    filters: Record<string, FilterValue | null>,
    sorter: SorterResult<T> | SorterResult<T>[]
  ) => void;
  emptyText?: string;
  size?: "small" | "middle" | "large";
  scroll?: { x?: number | string; y?: number | string };
  bordered?: boolean;
}

function DataTable<T extends object>({
  columns,
  dataSource,
  loading = false,
  pagination,
  rowKey = "id",
  rowSelection,
  onChange,
  emptyText = "No data available",
  size = "middle",
  scroll,
  bordered = false,
}: DataTableProps<T>) {
  return (
    <Table<T>
      columns={columns}
      dataSource={dataSource}
      loading={loading}
      pagination={
        pagination !== false
          ? {
              showSizeChanger: true,
              showTotal: (total, range) =>
                `${range[0]}-${range[1]} of ${total} items`,
              ...pagination,
            }
          : false
      }
      rowKey={rowKey}
      rowSelection={rowSelection}
      onChange={onChange}
      locale={{
        emptyText: <Empty description={emptyText} />,
      }}
      size={size}
      scroll={scroll}
      bordered={bordered}
    />
  );
}

export default DataTable;
