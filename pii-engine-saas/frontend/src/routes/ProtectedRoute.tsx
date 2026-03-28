import React from "react";
import { Navigate, useLocation } from "react-router-dom";
import { useAuthStore } from "@/store/authStore";

interface ProtectedRouteProps {
  children: React.ReactNode;
}

const ProtectedRoute: React.FC<ProtectedRouteProps> = ({ children }) => {
  const isAuthenticated = useAuthStore((state) => state.isAuthenticated);
  const accessToken = useAuthStore((state) => state.accessToken);
  const location = useLocation();

  // Check localStorage directly as the primary auth check -- this is
  // synchronous and available immediately, unlike Zustand persist hydration
  // which is async and can cause a brief unauthenticated flash.
  const hasLocalStorageToken = !!localStorage.getItem("pii_access_token");

  if (!hasLocalStorageToken && !isAuthenticated && !accessToken) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  return <>{children}</>;
};

export default ProtectedRoute;
