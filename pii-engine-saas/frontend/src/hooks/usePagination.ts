import { useState, useMemo, useCallback } from "react";

interface UsePaginationOptions {
  initialPage?: number;
  initialPageSize?: number;
}

interface UsePaginationReturn {
  page: number;
  pageSize: number;
  total: number;
  totalPages: number;
  setPage: (page: number) => void;
  setPageSize: (size: number) => void;
  setTotal: (total: number) => void;
  resetPage: () => void;
  paginationProps: {
    current: number;
    pageSize: number;
    total: number;
    showSizeChanger: boolean;
    showTotal: (total: number, range: [number, number]) => string;
    onChange: (page: number, pageSize: number) => void;
  };
}

export function usePagination(
  options: UsePaginationOptions = {}
): UsePaginationReturn {
  const { initialPage = 1, initialPageSize = 20 } = options;
  const [page, setPage] = useState(initialPage);
  const [pageSize, setPageSizeState] = useState(initialPageSize);
  const [total, setTotal] = useState(0);

  const totalPages = useMemo(
    () => Math.ceil(total / pageSize) || 1,
    [total, pageSize]
  );

  const resetPage = useCallback(() => setPage(1), []);

  const setPageSize = useCallback(
    (size: number) => {
      setPageSizeState(size);
      setPage(1);
    },
    []
  );

  const paginationProps = useMemo(
    () => ({
      current: page,
      pageSize,
      total,
      showSizeChanger: true,
      showTotal: (total: number, range: [number, number]) =>
        `${range[0]}-${range[1]} of ${total} items`,
      onChange: (newPage: number, newPageSize: number) => {
        if (newPageSize !== pageSize) {
          setPageSizeState(newPageSize);
          setPage(1);
        } else {
          setPage(newPage);
        }
      },
    }),
    [page, pageSize, total]
  );

  return {
    page,
    pageSize,
    total,
    totalPages,
    setPage,
    setPageSize,
    setTotal,
    resetPage,
    paginationProps,
  };
}
